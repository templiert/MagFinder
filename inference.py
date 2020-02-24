from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2 as cv
import os
import pathlib
import numpy as np
from collections import namedtuple
import json
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__ + ".Inference")
log.setLevel(logging.DEBUG)

Candidate = namedtuple('Candidate', 'contour label')


class Inference:
    """ Inference

    Inference is the part of the machine learning pipeline that is in charge of performing prediction 
    on a original-size wafer image. In order to do that, the original image is cut into several subimages 
    usually with the same dimensions as the patch dimensions used in the artificial patch generator. 
    Then, on each of the subimages, the prediction is performed using the predictor class, implemented in 
    predictor.py. The result of the prediction is the list of contours and its labels (tissue or magnet) 
    which we call candidates. The candidates will be used in the next step of the pipeline in order to 
    merge the prediction results on the full wafer. 
    
    Parameters
    ----------
    im: np.ndarray
        an original wafer image
    im_fluo: np.ndarray
        an fluorescent wafer image
    subimage_size: tuple
        dimension of the cuts of the original image, it has the same size as the patch used in the training
    overlap: int
        number of pixels overlap 
    config_file: str
        configuration file used for inference
    """

    def __init__(self, im, im_fluo, subimage_size, overlap, config_file):
        self.config_file = config_file        
        self.im = im
        self.im_fluo = im_fluo
        self.subimage_size = subimage_size
        self.overlap = overlap


    def collect_candidates(self, debug=False):
        """ Collect the candidates on the original wafer image
        
        Parameters
        ----------
        debug: bool
            True if we want to plot the steps of the inference. By default it is False.

        Returns
        -------
        candidates: list
            list of candidates that have their label (tissue or magnet) and the contour points
        """
        cfg.merge_from_file(self.config_file)
        
        coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.0)

        merged = np.zeros((self.im.shape[0], self.im.shape[1], 3), dtype=np.uint8)
        merged[:,:,0] = self.im
        merged[:,:,1] = self.im_fluo

        patchSizeEffective = self.subimage_size - self.overlap
        
        nXPatches = int(self.im.shape[0]/patchSizeEffective[1]) + 1
        nYPatches = int(self.im.shape[1]/patchSizeEffective[0]) + 1
        
        candidates = []
        iteration = 0 
        total_iterations = nXPatches * nYPatches
        for idx in range(nXPatches):
            for idy in range(nYPatches):
                iteration += 1
                patch = np.zeros((self.subimage_size[1], self.subimage_size[0], 3), dtype=np.uint8)

                x1 = idx * patchSizeEffective[1]
                x2 = min(x1 + self.subimage_size[1], self.im.shape[0])

                y1 = idy * patchSizeEffective[0]
                y2 = min(y1 + self.subimage_size[0], self.im.shape[1])

                extract = merged[x1:x2, y1:y2,:]
                patch[:extract.shape[0], :extract.shape[1],:] = extract

                if iteration%10 == 0:
                    print(f'[{iteration}/{total_iterations}] processing image')

                result = coco_demo.run_on_opencv_image(patch)

                x, y = y1, x1  
                patchOffset = np.array([x,y])

                for contour, label in zip(result.contours, result.labels):
                    points = np.array([c[0] + patchOffset for c in contour])
                    candidate = Candidate(contour=points, label=label)
                    candidates.append(candidate)

                    if debug:
                        # show image and overlay
                        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
                        ax[0].imshow(patch)
                        ax[0].set_title('Processing Image')
                        ax[1].imshow(result.overlay)
                        ax[1].set_title('Result Overlay')
                        plt.show()

        log.info(f"Number of candidates {len(candidates)}")
        log.info(f"Number of patches in x and y dimensions {nXPatches}, {nYPatches}")

        return candidates