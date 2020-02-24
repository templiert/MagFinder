import os
import torch
import numpy as np
import pathlib
import argparse
import math
import logging

from shutil import copy
from template_stats import TemplateStats
from trim_detectron_model import TrimDetectronModel
from artificial_patch_generator import ArtificialPatchGenerator
from misc import read_input_data, resize_input
from inference import Inference
from wafer_merge import WaferMerge
from misc import get_config, set_config
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog


# taken from Wafer 17 template stats
DESIRABLE_DISTANCE_INSIDE_CONTOUR = 76

# modify learning rate based on https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml
DICT_NGPUS_AND_LR_RATE = {1:0.005, 2:0.01, 4:0.02}

PIPELINE_SETUP_DIR = os.path.join(os.getcwd(), "PIPELINE_SETUP")

# this artificial folder path is added in maskrcnn_benchmark/config/paths_catalog.py
ARTIFICIAL_FOLDER = os.path.join(os.getcwd(), "datasets", "artificial_folder")
pathlib.Path(ARTIFICIAL_FOLDER).mkdir(parents=True, exist_ok=True)

log = None

def setup_logging(wafer_dir):
    global log
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(wafer_dir, "log_pipeline.txt"), "a+")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)


def pipeline(wafer_dir, ngpus=None, num_patches=None, num_throws=None, num_angles=None, ratio=None,
             patch_size=None, subimage_size=None, area_threshold=None, min_distance_threshold=None, keep=False):
    """ Machine Learning Pipeline for Section Segmentation
    
    This ML pipeline goes through the several steps to detect sections containing brain tissue on a silicon wafer.
    Steps are following:
    - trimming detectron model
    - artificial patch generator
    - training
    - inference
    - wafer merge.
    First, we start by preparing the files for the pipeline. Afterwards, we go step-by-step through the pipeline.

    Parameters
    ----------
    wafer_dir: str 
        Path to the directory containing all the wafer information needed for the pipeline (2 *.tif images and 1 json labelme file).
    ngpus: int
        Number of GPUs to be used in the training. If None, it will get the number of available GPUs using torch library. 
    num_patches: int
        Number of patches used for the artificial patch generator. If None, it will use the settings specified in the pipeline_config.yaml. 
    num_throws: int
        Number of throws on the patch during the artificial pathc generator. If None, it will use the settings specified in the pipeline_config.yaml. 
    num_angles: int
        Number of points (angles) in the 0 to 360 degrees range. If None, it will use the settings specified in the pipeline_config.yaml. 
    ratio: float
        Train/test split ratio used for the training. If None, it will use the settings specified in the pipeline_config.yaml.   
    patch_size: str tuple
        Size of the patch that will be generated in the artificial pathc generator. If None, it will use the settings specified in the pipeline_config.yaml. 
        Input is a string in a shape of tuple, e.g. '(400,400)'
    subimage_size: str tuple
        Size of the subimage that will be used in inference when cutting original image to the smaller subimages. If None, it will use the settings specified in the pipeline_config.yaml. 
        Input is a string in a shape of tuple, e.g. '(400,400)'
    area_threshold: float
        Area threshold that will be used in inference to filter out the sections based on this area threshold. It is in range from [0, 1] and it will be applied to the mean area size of the class. 
        If None, it will use the settings specified in the pipeline_config.yaml. 
    min_distance_threshold: float
        Min distance threshold that will be used in inference to filter out the sections during the clustering. It is in range from [0, 1] and it will be applied to the mean distance of the class. 
        If None, it will use the settings specified in the pipeline_config.yaml. 
    keep: bool
        Define if you want to keep the Tran and Test data used for the training. If False, it won't save the directories by default, otherwise it will.
    """
    setup_logging(wafer_dir)

    log.info("Prepare the files for the pipeline")

    pipeline_config_path = os.path.join(wafer_dir, "pipeline_config.yaml")
    if not os.path.isfile(pipeline_config_path):
        pipeline_config_path = copy(os.path.join(PIPELINE_SETUP_DIR, "pipeline_config.yaml"), pipeline_config_path)
    log.info(f"Read the pipeline configuration, path: {pipeline_config_path}")

    pipeline_config = get_config(pipeline_config_path, printing=True)
    pipeline_config = pipeline_config["PIPELINE"]

    pretrained_model_path = os.path.join(wafer_dir, "model_final.pkl")
    if not os.path.isfile(pretrained_model_path):
        pretrained_model_path = os.path.join(PIPELINE_SETUP_DIR, pipeline_config["TRIMMING"]["PRETRAINED_MODEL"])
    pipeline_config["TRIMMING"]["PRETRAINED_MODEL"] = pretrained_model_path
    log.info(f"Read the pretrained model for trimming, path: {pretrained_model_path}")

    configuration_path = os.path.join(wafer_dir, "e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml")
    if not os.path.isfile(configuration_path):
        configuration_path = copy(os.path.join(PIPELINE_SETUP_DIR, pipeline_config["TRIMMING"]["CONFIGURATION"]), configuration_path)
    pipeline_config["TRIMMING"]["CONFIGURATION"] = configuration_path
    log.info(f"Read the training configuration, path: {configuration_path}")

    trimmed_model_path = os.path.join(PIPELINE_SETUP_DIR, "trimmedModel_X101.pth")
    log.info(f"Define saving path for trimmed model, path: {trimmed_model_path}")

    labelme_template_path = os.path.join(wafer_dir, "init_labelme.json")
    if not os.path.isfile(labelme_template_path):
        raise RuntimeError(f"File {labelme_template_path} does not exist!")
    log.info(f"Read init_labelme.json initial template file, path: {labelme_template_path}")

    original_image_path = os.path.join(wafer_dir, "wafer.tif")
    if not os.path.isfile(original_image_path):
        raise RuntimeError(f"File {original_image_path} does not exist!")
    log.info(f"Read wafer.tif original wafer image, path: {original_image_path}")

    fluo_image_path = os.path.join(wafer_dir, "wafer_fluo.tif")
    if not os.path.isfile(fluo_image_path):
        raise RuntimeError(f"File {fluo_image_path} does not exist!")
    log.info(f"Read wafer_fluo.tif fluorescent wafer image, path: {fluo_image_path}")

    labelme_init, im, im_fluo = read_input_data(labelme_template_path, original_image_path, fluo_image_path)

    ngpus = ngpus or torch.cuda.device_count()    
    pipeline_config["TRAINING"]["NGPUS"] = ngpus
    log.info(f"Number of gpus for training: {ngpus}")

    # artificial patch generator
    num_patches = num_patches or pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_PATCHES"]  
    pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_PATCHES"] = num_patches
    log.info(f"Number of pathces: {num_patches}")

    num_throws = num_throws or pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_THROWS"]   
    pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_THROWS"] = num_throws
    log.info(f"Number of throws: {num_throws}")

    num_angles = num_angles or pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_ANGLES"] 
    pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["NUM_ANGLES"] = num_angles
    log.info(f"Number of angles: {num_angles}")

    ratio = ratio or pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["RATIO"] 
    pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["RATIO"] = ratio
    log.info(f"Train-test split ratio: {ratio}")

    patch_size = patch_size or pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["PATCH_SIZE"]	
    pipeline_config["ARTIFICIAL_PATCH_GENERATOR"]["PATCH_SIZE"] = patch_size
    patch_size = np.array(list(eval(patch_size)))
    log.info(f"Patch size is: {patch_size}")

    # inference
    inference_config_file = os.path.join(wafer_dir, "inferenceConfigX101.yaml")
    if not os.path.isfile(inference_config_file):
        inference_config_file = copy(os.path.join(PIPELINE_SETUP_DIR, pipeline_config["INFERENCE"]["CONFIGURATION"]), inference_config_file)
    pipeline_config["INFERENCE"]["CONFIGURATION"] = inference_config_file
    log.info(f"Read inference configuration file, path: {inference_config_file}")
    
    subimage_size = subimage_size or pipeline_config["INFERENCE"]["SUBIMAGE_SIZE"]
    subimage_size = np.array(list(eval(subimage_size)))
    log.info(f"Subimage size is: {subimage_size}")

    # wafer merge
    area_threshold = area_threshold or pipeline_config["WAFER_MERGE"]["AREA_THRESHOLD"] 
    pipeline_config["WAFER_MERGE"]["AREA_THRESHOLD"] = area_threshold
    log.info(f"Area threshold is: {area_threshold}")

    min_distance_threshold = min_distance_threshold or pipeline_config["WAFER_MERGE"]["MIN_DISTANCE_THRESHOLD"]
    pipeline_config["WAFER_MERGE"]["MIN_DISTANCE_THRESHOLD"] = min_distance_threshold
    log.info(f"Min distance threshold is: {min_distance_threshold}")

    output_labelme = os.path.join(wafer_dir, 'output.json')
    log.info(f"Output file will be in: {output_labelme}")

    # modify training and inference configuration files
    ts = TemplateStats()
    ts.collect(labelme_init=labelme_init)

    train_config = get_config(configuration_path, printing=False)
    inference_config = get_config(inference_config_file, printing=False)
    
    # change learning rate
    train_config["SOLVER"]["BASE_LR"] = DICT_NGPUS_AND_LR_RATE.get(ngpus, 0.02)
    inference_config["SOLVER"]["BASE_LR"] = DICT_NGPUS_AND_LR_RATE.get(ngpus, 0.02)
    log.info(f"Change learning rate based on number of GPUs: {DICT_NGPUS_AND_LR_RATE.get(ngpus, 0.02)}")

    set_config(train_config, configuration_path, printing=False)
    set_config(inference_config, inference_config_file, printing=False)

    # resize image
    resize_factor = DESIRABLE_DISTANCE_INSIDE_CONTOUR/ts.distance_inside_contour
    labelme_init, im, im_fluo = resize_input(labelme_init, im, im_fluo, resize_factor)

    # need to collect new stats
    ts = TemplateStats()
    ts.collect(labelme_init=labelme_init)

    ### START THE PIPELINE ###
    log.info("Start the pipeline")

    log.info("Start trimming detectron model")
    tdm = TrimDetectronModel(pretrained_model_path=pretrained_model_path, configuration_path=configuration_path, trimmed_model_path=trimmed_model_path)
    tdm.trim()
    log.info("End trimming detectron model")

    log.info("Start artificial pathc generator")
    apg = ArtificialPatchGenerator(im=im, im_fluo=im_fluo, labelme_init=labelme_init, artificial_folder=ARTIFICIAL_FOLDER, patch_size=patch_size, 
                                   num_patches=num_patches, num_throws=num_throws, num_angles=num_angles, ratio=ratio)
    apg.create_train_and_test_data()
    log.info("End artificial pathc generator")

    log.info("Empty wafer dir's artificial data")
    apg.delete_train_and_test_data(os.path.join(wafer_dir, "artificial_folder"))
    if keep:
        log.info("Save Train/Test in wafer dir as well")
        apg.keep_artificial_data_in_wafer_dir(wafer_dir)

    log.info("Start training")
    os.system(f"python -m torch.distributed.launch --nproc_per_node={ngpus} tools/train_net.py --config-file={configuration_path}")
    log.info("End training")

    apg.delete_train_and_test_data()
    log.info("Remove Train/Test data from artificial folder")

    log.info("Start inference")
    inference = Inference(im, im_fluo, subimage_size=subimage_size, overlap=ts.overlap, config_file=inference_config_file) 
    candidates = inference.collect_candidates()
    log.info("End inference")

    log.info("Start wafer merge")
    wm = WaferMerge(im, template_stats=ts, candidates=candidates)
    data = wm.finding_best_candidates(area_threshold, min_distance_threshold, ratio=1/resize_factor)
    wm.save_to_file(data, output_labelme)
    log.info("End wafer merge")

    log.info("End the pipeline")
 
    new_pipeline_config = {}
    new_pipeline_config["PIPELINE"] = pipeline_config
    set_config(new_pipeline_config, os.path.join(wafer_dir, "pipeline_config.yaml"), printing=False)
    log.info("Overwrite the settings or just create the new one")

    if os.path.exists("last_checkpoint"):
        os.remove("last_checkpoint") 
    log.info("Remove last_checkpoint from the repo")

    if os.path.exists("model_final.pth"):
        os.remove("model_final.pth")
    log.info("Remove model_final.pth from the repo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline for Brain Segmentation')

    parser.add_argument('--wafer-dir', help='Directory that contains all necessary data of the wafer',
                        required=True, type=str)
    parser.add_argument('--ngpus', help='Number of GPUs used for training. Power of 2.',
                        required=False, default=None, type=int)
    parser.add_argument('--num-patches', help='Number of patches to generate with artificial patch generator',
                        required=False, default=None, type=int)
    parser.add_argument('--num-throws', help='Number of throws used in artificial patch generator',
                        required=False, default=None, type=int)
    parser.add_argument('--num-angles', help='Number of angles to rotate generated sections from 0 to 360 degrees in artificial patch generator',
                        required=False, default=None, type=int)
    parser.add_argument('--ratio', help='Ratio of split data into training and validation sets',
                        required=False, default=None, type=int)
    parser.add_argument('--patch-size', help='Patch sizes to be generated in artificial patch generator, eg. (400,400)',
                        required=False, default=None, type=str)
    parser.add_argument('--subimage-size', help='Subimage sizes used in inference, eg. (400,400)',
                        required=False, default=None, type=str)
    parser.add_argument('--area-limit', help='Area threshold to use when merging wafer',
                        required=False, default=None, type=int)
    parser.add_argument('--min-distance-limit', help='Min distance threshold to use when merging wafer',
                        required=False, default=None, type=int)
    parser.add_argument('--keep', help='Keep artificially generated data in wafer-dir/artificial_folder', 
                        action='store_true')

    args = parser.parse_args()

    pipeline(wafer_dir=args.wafer_dir, ngpus=args.ngpus, num_patches=args.num_patches, 
        num_throws=args.num_throws, num_angles=args.num_angles, ratio=args.ratio,
        patch_size=args.patch_size, subimage_size=args.subimage_size, 
        area_threshold=args.area_limit, min_distance_threshold=args.min_distance_limit,
        keep=args.keep)
