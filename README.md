
# Machine Learning Pipeline

Software tools for machine learning pipeline used for detection of brain tissue on a silicon wafers.

- SSH the VM
- Manually clone the repo: 
``` 
git clone https://github.com/templiert/MagFinder
```
## Docker
### Build Docker (to do only once per machine at installation)

```
sudo nvidia-docker build --build-arg USERNAME=YOUR_USERNAME --build-arg PASSWORD=YOUR_PASSWORD -t sectionsegmentation-app:latest .
```

### Docker Local (for the VM - needs to be run at each new startup of the VM)
```
sudo nvidia-docker run --device /dev/fuse --cap-add SYS_ADMIN --shm-size=8G -it sectionsegmentation-app
```
### Docker Website (for the Kubernete)
```
sudo nvidia-docker run -p 5000:5000 --device /dev/fuse --cap-add SYS_ADMIN --shm-size=8G -it sectionsegmentation-app
```

## Preparation for the Pipeline
Before runing the pipeline, you should prepare the wafer directory. This directory contains all the information regarding the wafer on which we want to discover the section coordinates.

The example of the structure of the directory is the following:
```
wafer-dir/
  init_labelme.json [*]
  wafer.tif [*]
  wafer_fluo.tif [*]
  model_final.pkl
  e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml
  inferenceConfigX101.yaml
```
Note: 
- the names should be exactly the same as above, otherwise, they will not be recognized from the script. 
- the [*] represents files that are required, others are optional and will be loaded from the default settings in case they are not specified.

## Run the Pipeline

To run the pipeline, you can do it the following way:

- run the pipeline
```
python pipeline.py --wafer-dir /path/to/wafer-dir/
```
Important note: to be sure that the job terminates even if you disconnect your ssh and/or close your terminal, use 
```
nohup xxx &
```
the following way:

```
nohup python pipeline.py --wafer-dir /path/to/wafer-dir/ &
```

Note: there is an example directory in the repo, so you can run your first pipeline with the following command:
```
python pipeline.py --wafer-dir=wafer-dir-example/
```


More information about the other parameters can be found bellow:
```
usage: pipeline.py [-h] --wafer-dir WAFER_DIR [--ngpus NGPUS]
               [--num-patches NUM_PATCHES] [--num-throws NUM_THROWS]
               [--num-angles NUM_ANGLES] [--ratio RATIO]
               [--patch-size PATCH_SIZE] [--area-limit AREA_LIMIT]
               [--min-distance-limit MIN_DISTANCE_LIMIT]

Machine Learning Pipeline for Brain Segmentation

optional arguments:
  -h, --help            show this help message and exit
  --wafer-dir WAFER_DIR
                        Directory that contains all necessary data of the
                        wafer
  --ngpus NGPUS         Number of GPUs used for training
  --num-patches NUM_PATCHES
                        Number of patches to generate with artificial patch
                        generator
  --num-throws NUM_THROWS
                        Number of throws used in artificial patch generator
  --num-angles NUM_ANGLES
                        Number of angles to rotate generated sections from 0
                        to 360 degrees in artificial patch generator
  --ratio RATIO         Ratio of split data into training and validation sets
  --patch-size PATCH_SIZE
                        Patch sizes to be generated in artificial patch
                        generator
  --area-limit AREA_LIMIT
                        Area threshold to use when merging wafer
  --min-distance-limit MIN_DISTANCE_LIMIT
                        Min distance threshold to use when merging wafer
  --keep KEEP
						Keep artificially generated data in wafer-dir/artificial_folder. 
                        Define if you want to keep the Tran and Test data used for the 
                        training. If False, it won't save the directories by default, 
                        otherwise it will.
```

In the `sectionsegmentationml/PIPELINE_SETUP` all the default settings for the pipeline are located, including configuration files, initial detectron model, and pipeline configuration file. Therefore, in case some of the provided input parameters are not set by user, they will be read from that directory.

Note: when modifying the code in the repo, you need to pull the updated repo from inside the repo

## Kubernetes
For kubernetes, please see directory kubernetes/