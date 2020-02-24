import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
import logging

log = logging.getLogger(__name__ + ".TrimDetectronModel")
log.setLevel(logging.DEBUG)


class TrimDetectronModel:
    """ Trimming Detectron Model
    
    Since transfer learning is particularly useful for rapid progress and improved performance, we will 
    use pre-trained models from Detectron. 
    Detectron is Facebook AI Research's software system that implements state-of-the-art object detection 
    algorithms, including Mask R-CNN. The goal of Detectron is to provide a flexible codebase to support 
    rapid implementation and evaluation of object detection research. In the Detectron github repository 
    they provide a large set of trained models available for 
    download (https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). 
    Detectron uses the backbone models pretrained on ImageNet and 
    models are trained on the COCO dataset. 
    The Mask R-CNN github repository also provides the set of 
    baselines (https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/MODEL_ZOO.md) 
    which are trained using the same setup as Detectron. For this project, the pre-trained model with 
    backbone X-101-32x8d-FPN is used.

    After the pre-trained model is downloaded, the first step of the machine learning pipeline has 
    the goal to finetune Detectron weights on custom datasets, as described in the Mask R-CNN 
    repository (https://github.com/facebookresearch/maskrcnn-benchmark\#finetuning-from-detectron-weights-on-custom-datasets). 
    Since the last layers of this pre-trained model have classes that are different (e.g. cat, dog, etc.) from the classes 
    we have in our dataset (tissue, magnet), we should not load these weights. This script removes keys corresponding to the last layer. 
    Trimmed model is stored and ready to be used as a starting point in the Mask R-CNN training on our dataset.

    Parameters
    ----------
    pretrained_model_path: str
        path to the pretrained Detectron model
    configuration_path: str
        configuration file used in the pretrained model
    trimmed_model_path: str
        path where the trimmed model will be stored
    """

    def __init__(self, pretrained_model_path, configuration_path, trimmed_model_path):
        self.pretrained_model_path = pretrained_model_path
        self.configuration_path = configuration_path
        self.trimmed_model_path = trimmed_model_path

    def _removekey(self, d, listofkeys):
        """ Remove key from the dictionary

        Removes the keys representing the last layers of the network

        Parameters
        ----------
        d: dict
            dictionary that will have values popped
        listofkeys: list
            list of keys that will be removed from the dictionary 
        """
        log.debug(f"list of keys to remove: {listofkeys}")
        r = dict(d)
        for key in listofkeys:
            log.info(f'key: {key} is removed')
            r.pop(key)
        return r

    def trim(self):
        """ Main method for trimming the Detectron model       
        """
        DETECTRON_PATH = os.path.expanduser(self.pretrained_model_path)
        log.info(f'Path to pretrained Detecron model: {DETECTRON_PATH}')

        cfg.merge_from_file(self.configuration_path)
        trimmed_model = load_c2_format(cfg, DETECTRON_PATH)

        #log.debug('Pre-trained model keys are: \n\t','\n\t'.join(trimmed_model['model'].keys()))
        trimmed_model['model'] = self._removekey(trimmed_model['model'],
                                    ['cls_score.bias', 
                                    'cls_score.weight', 
                                    'bbox_pred.bias', 
                                    'bbox_pred.weight', 
                                    'mask_fcn_logits.bias', 
                                    'mask_fcn_logits.weight'])
        torch.save(trimmed_model, self.trimmed_model_path)
        log.info(f'Trimmed model saved to: {self.trimmed_model_path}')

