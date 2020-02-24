import json
import os
import random
import pathlib
import copy
import sys
import numpy as np
import cv2 as cv
import logging
from shutil import rmtree
from distutils.dir_util import copy_tree

log = logging.getLogger(__name__ + ".ArtificialPatchGenerator")
log.setLevel(logging.DEBUG)


class ArtificialPatchGenerator:
    """ Artificial Patch Generator

    The goal of artificial patch generator is to create a high amount of images of different sizes 
    that will be used in the training. These images are created from the ML pipeline input data. 
    To make use of both input images we have, we combine them into 3-channel images. 
    The first channel contains the original image; the second channel contains fluorescent image; and 
    the third channel is empty. 

    For example, let's say that user annotated five sections and one background in the input template file. 
    The artificial patch generator will use the background information to create the background of the patch. 
    Also, it will use the information of five previously annotated different sections to generate new ones 
    by rotating known ones and randomly throwing them on the patch background. This way of augmenting data 
    is the reason why it is important to initially annotate as many different looking sections as possible. 

    im: np.ndarray
        an original wafer image 
    im_fluo: np.ndarray
        an fluorescent wafer image
    labelme_init: dict
        containing manually labeled sections from the user
    artificial_folder: str 
        the path to the artificial folder where data will be generated
    patch_size: tuple
        patch size that will be generated for the training, e.g. (400, 400)
    num_patches: int 
        number of the patches to be generated
    num_throws: int
        number of throws on the patch during the patch generation
    num_angles: int 
        number of angles from 0 to 360 that sections will be rotated and placed on the patch
    ratio: float
        ration (between 0 and 1) represents the split between training and validation data
    """

    def __init__(self, im, im_fluo, labelme_init, artificial_folder, patch_size, num_patches, num_throws, num_angles, ratio):
        self.im = im
        self.im_fluo = im_fluo
        self.labelme_init = labelme_init

        self.artificial_folder = artificial_folder

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_throws = num_throws
        self.num_angles = num_angles
        self.ratio = ratio


    def _get_templates(self): 
        """ Get templates for artificial data generation

        Collect templates that will be used during the creation of train and validation datasets. Based on 
        this information, sections on the patches will be transformed (rotated) and placed on the patch.
        Here, the sections are rotated and collected in the templates dictionary.

        The output template dictionary looks like:

        templates ={
            1: {
                'bbox': np.ndarray(...),           
                'e': {                               # envelope
                    'mask': np.ndarray(...),
                    'points': np.ndarray(...),
                    'im': masked image,
                    'fluo': masked fluo image
                },
                'eSize': cv.countNonZero(...),

                't': {                               # tissue
                    'mask': np.ndarray(...),
                    'points': np.ndarray(...)
                },
                'tSize': cv.countNonZero(...),

                'm': {                               # magnet
                    'mask': np.ndarray(...),
                    'points': np.ndarray(...)
                },
                'mSize': cv.countNonZero(...),
            },
            2: {
                ...
            }
        }

        Parameters
        ----------
        templates: dict
            Template containing information of sections that were manually labeled by the user.

        """
        sections = list(zip(*(iter(self.labelme_init['shapes'][:-1]),) * 3))

        templates = {}

        angles = np.linspace(start=0, stop=360, num=self.num_angles)[:-1]
        counter = 0

        for idx, [t, m, e] in enumerate(sections):
            log.info(f"Section template No.{idx + 1} (contains tissue [t], magnet [m], envelope [e])")

            for angle in angles:
                log.debug("Get a larger bbox around the template to make rotations")
                bbox = cv.boundingRect(np.array(e['points']))
                log.debug(f"Bounding box of the envelope {bbox}")
                largerBbox = np.array([bbox[0]-bbox[2], bbox[1]-bbox[3], 3*bbox[2], 3*bbox[3]])

                log.debug("Check that the larger bbox is still in the image")
                if largerBbox[0] > 0 and largerBbox[1] > 0 and largerBbox[0] + largerBbox[2] < self.im.shape[0] and largerBbox[1] + largerBbox[3] < self.im.shape[1]: 
                    templates[counter] = {}

                    M = cv.getRotationMatrix2D((largerBbox[2]/2, largerBbox[3]/2), angle, 1)
                    log.debug(f"Rotation matrix {M}")

                    # envelope
                    log.debug("Process Envelope")
                    ePoints = np.array(e['points']) - np.array([largerBbox[0], largerBbox[1]])
                    ePointsRotated = cv.transform(np.array([ePoints]), M)[0]

                    bboxAfterRotation = cv.boundingRect(np.array(ePointsRotated))
                    bboxAfterRotation = [bboxAfterRotation[0] - 1, bboxAfterRotation[1] - 1, bboxAfterRotation[2] + 2, bboxAfterRotation[3] + 2]
                    templates[counter]['bbox'] = bboxAfterRotation

                    eMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8)
                    cv.fillConvexPoly(eMask, ePointsRotated, 255)
                    eMask = eMask[bboxAfterRotation[1]:bboxAfterRotation[1] + bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0] + bboxAfterRotation[2]]

                    templates[counter]['e'] = {}
                    templates[counter]['e']['mask'] = eMask
                    templates[counter]['e']['points'] = np.array(ePointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]]) + np.array([0,0])
                    templates[counter]['eSize'] = cv.countNonZero(eMask)

                    # tissue
                    log.debug("Process Tissue")
                    tPoints = np.array(t['points']) - np.array([largerBbox[0], largerBbox[1]])
                    tPointsRotated = cv.transform(np.array([tPoints]), M)[0]

                    tMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8)
                    cv.fillConvexPoly(tMask, tPointsRotated, 255)
                    tMask = tMask[bboxAfterRotation[1]:bboxAfterRotation[1] + bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0] + bboxAfterRotation[2]]

                    templates[counter]['t'] = {}
                    templates[counter]['t']['points'] = np.array(tPointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]]) + np.array([0,0])
                    templates[counter]['t']['mask'] = tMask
                    templates[counter]['tSize'] = cv.countNonZero(tMask)

                    # magnet
                    log.debug("Process Magnet")
                    mPoints = np.array(m['points']) - np.array([largerBbox[0], largerBbox[1]])
                    mPointsRotated = cv.transform(np.array([mPoints]), M)[0]

                    mMask = np.zeros((largerBbox[3], largerBbox[2]), dtype=np.uint8)
                    cv.fillConvexPoly(mMask, mPointsRotated, 255)
                    mMask = mMask[bboxAfterRotation[1]:bboxAfterRotation[1] + bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0] + bboxAfterRotation[2]]

                    templates[counter]['m'] = {}
                    templates[counter]['m']['points'] = np.array(mPointsRotated) - np.array([bboxAfterRotation[0], bboxAfterRotation[1]]) + np.array([0,0])
                    templates[counter]['m']['mask'] = mMask
                    templates[counter]['mSize'] = cv.countNonZero(mMask)

                    # process the images
                    log.debug("Process the Images")
                    imBoxed = self.im[largerBbox[1]:largerBbox[1] + largerBbox[3], largerBbox[0]:largerBbox[0] + largerBbox[2]]
                    fluoBoxed = self.im_fluo[largerBbox[1]:largerBbox[1] + largerBbox[3], largerBbox[0]:largerBbox[0] + largerBbox[2]]

                    imRotated = cv.warpAffine(imBoxed, M, (largerBbox[2], largerBbox[3]))
                    fluoRotated = cv.warpAffine(fluoBoxed, M, (largerBbox[2], largerBbox[3]))

                    imCropped = imRotated[bboxAfterRotation[1]:bboxAfterRotation[1] + bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0] + bboxAfterRotation[2]]
                    imMasked = cv.bitwise_and(imCropped, imCropped, mask = eMask)
                    templates[counter]['e']['im'] = imMasked

                    fluoCropped = fluoRotated[bboxAfterRotation[1]:bboxAfterRotation[1] + bboxAfterRotation[3], bboxAfterRotation[0]:bboxAfterRotation[0] + bboxAfterRotation[2]]
                    fluoMasked = cv.bitwise_and(fluoCropped, fluoCropped, mask = eMask)
                    templates[counter]['e']['fluo'] = fluoMasked

                    counter += 1

        return templates

    def _create_backgrounds(self):
        """ Create background of the patch

        This method will use the background information that was labeled in the initial template JSON labelme file that 
        user provided as an input. Depending on the size of the labeled background, it will create a background for the patch.
        If the labeled background is bigger than the specified patch size, this background will be used.
        If the labeled background has smaller dimensions than the specified patch size, it will generate
        the background that has uniform color (the color will be calculated as a mean value of labeled background).

        Returns
        -------
        backgrounds: list
            List of background for original wafer patch
        backgroundsFluo: list
            List of backgrounds for fluorescent wafer patch
        """
        backgrounds = []
        backgroundsFluo = []

        backgroundPoints = np.array(self.labelme_init['shapes'][-1]['points'])
        backgroundBbox = cv.boundingRect(backgroundPoints)

        if backgroundBbox[2] > self.patch_size[0] and backgroundBbox[3] > self.patch_size[1]:
            log.info(f"Template background size {backgroundBbox} is big enough to create the patch of size {self.patch_size}")

            by = backgroundBbox[1] + backgroundBbox[3]
            bx = backgroundBbox[0] + backgroundBbox[2]
            log.debug("The labeled background is larger than the patchSize")
            
            log.info(f"Background x and y: {bx} {by}")
            shiftSize = (self.patch_size * 0.2).astype(int) 
            log.debug(f"Shift size represents how much does the window slide to generate the background (these are subpatches of the manually selected (large) background area)")
            
            log.debug("No need to bother with correct indices, just try 10x10 boxes and check whether they are in the image")
            for x in range(10): 
                for y in range(10):
                    y1 = backgroundBbox[1] + x*shiftSize[0]
                    x1 = backgroundBbox[0] + y*shiftSize[1]
                    x2 = x1 + self.patch_size[0]
                    y2 = y1 + self.patch_size[1]

                    log.debug(f"x2={x2} < bx{bx} and y2={y2} < by={by}")
                    if x2 < bx and y2 < by:
                        background = self.im[y1:y2, x1:x2]
                        backgrounds.append(background)

                        backgroundFluo = self.im_fluo[y1:y2, x1:x2]
                        backgroundsFluo.append(backgroundFluo)
        else:
            log.info(f"Template background size {backgroundBbox} is NOT big enough to create the patch of size {self.patch_size}. We create one color background!")
            
            # the labeled background is smaller than the patchSize, the background then simply has the dimensions patchSize and is uniform with the mean pixel intensity
            b1,g1,r1,_ = np.uint8(cv.mean(self.im[backgroundBbox[0]:backgroundBbox[0] + backgroundBbox[2], backgroundBbox[1]:backgroundBbox[1] + backgroundBbox[3]]))
            b2,g2,r2,_ = np.uint8(cv.mean(self.im_fluo[backgroundBbox[0]:backgroundBbox[0] + backgroundBbox[2], backgroundBbox[1]:backgroundBbox[1] + backgroundBbox[3]]))

            background = b1 * np.ones((self.patch_size[1], self.patch_size[0]), dtype=np.uint8)
            backgroundFluo = b2 * np.ones((self.patch_size[1], self.patch_size[0]), dtype=np.uint8)

            backgrounds.append(background)
            backgroundsFluo.append(backgroundFluo)

        if len(backgrounds) == 0:
                raise Exception("It couldn't collect the background. Please check the artificial_patch_generator.py implementation of creating backgrounds.")
                
        return backgrounds, backgroundsFluo

    def create_train_and_test_data(self):
        """ Create train and validation datasets for the training

        This method will create and store the directory with images (patches) as well as the annotation JSON file 
        containing the corresponding labels on those patches.
        This will be done for train dataset and validation dataset separatelly.

        The structure of artificial folder will contain:
        train/
            images/
                *.jpg
            coco_annotations.json
        val/
            images/
                *.jpg
            coco_annotations.json
        """
        
        templates = self._get_templates()
        backgrounds, backgroundsFluo = self._create_backgrounds()

        log.info("Creating training and validation data")
        dataset_types = ['train', 'val']
        ratios = [int(self.ratio*self.num_patches), self.num_patches-int(self.ratio*self.num_patches)]

        for idx, dataset_type in enumerate(dataset_types):
            imageFolder = os.path.join(self.artificial_folder, dataset_type, 'images')
            pathlib.Path(imageFolder).mkdir(parents=True, exist_ok=True)

            annotationsPath = os.path.join(self.artificial_folder, dataset_type, 'coco_annotations.json')
            if not os.path.exists(annotationsPath):
                categories = [{'id': 0, 'name': 'background', 'supercategory': 'label'},
                              {'id': 1, 'name': 'tissue', 'supercategory': 'label'},
                              {'id': 2, 'name': 'magnet', 'supercategory': 'label'}]
                
                data = {'images':[], 'annotations':[], 'categories':categories}
            else:
                data = json.load(open(annotationsPath, 'r'))
            
            annotationId = len(data["annotations"])
            patch_offset = len(next(os.walk(imageFolder))[2])

            log.debug("Main loop through the patches to be generated")
            for idPatch in range(ratios[idx]):
                adjustedIdPatch = patch_offset + idPatch 
                successfulThrows = []
                
                basket = np.zeros(self.patch_size[::-1], np.uint8) 
                log.debug(f"Container in which sections are thrown: {basket}")
                for idThrow in range(self.num_throws):
                    log.debug("Pick random template with a random orientation")
                    templateId = random.choice(list(templates.keys()))
                    template = templates[templateId]
                    bbox = template['bbox']
                    
                    throwSize = self.patch_size - np.array([bbox[2], bbox[3]]) 
                    log.debug(f"Thrown bounding boxes will be contained in the basket, throw size {throwSize}")
 
                    x = random.randint(0, throwSize[0])
                    y = random.randint(0, throwSize[1])
                    log.debug(f"Randomly throw x = {x}, y = {y}")

                    basketBox = basket[y:y + bbox[3], x:x + bbox[2]] # attention with x,y flip
                    log.debug(f"Extract the box in basket (the current image keeping track of the thrown envelopes)")

                    log.debug("Count the number of white pixels before and after throwing")
                    currentWhite = cv.countNonZero(basketBox)
                    basketBoxAfterThrow = cv.add(basketBox, template['e']['mask'])
                    whiteAfterThrow = cv.countNonZero(basketBoxAfterThrow)
                    if whiteAfterThrow - currentWhite == template['eSize']:
                        log.debug("No collision")
                        basket[y:y + bbox[3], x:x + bbox[2]] = basketBoxAfterThrow
                        successfulThrows.append([templateId, x, y])

                log.debug("Choose a background randomly")
                backgroundId = random.randint(0, len(backgrounds)-1)
                imPatch = copy.deepcopy(backgrounds[backgroundId])
                fluoPatch = copy.deepcopy(backgroundsFluo[backgroundId])


                log.debug("Create the images")
                for [templateId, x, y] in successfulThrows:
                    template = templates[templateId]
                    
                    tissuePoints = template['t']['points'] + np.array([x, y])
                    magPoints = template['m']['points'] + np.array([x, y])

                    polyTissue = [list(map(int, list(tissuePoints.ravel().astype(int))))]
                    polyMag = [list(map(int, list(magPoints.ravel().astype(int))))]

                    bboxTissue = list(map(int, list(cv.boundingRect(tissuePoints))))
                    bboxMag = list(map(int, list(cv.boundingRect(magPoints))))

                    annotation = {'segmentation': polyTissue, 
                                  'area': int(template['tSize']), 
                                  'iscrowd':0, 
                                  'image_id':int(adjustedIdPatch), 
                                  'bbox':bboxTissue, 
                                  'category_id': 1, 
                                  'id': int(annotationId)}
                    annotationId += 1
                    data["annotations"].append(annotation)

                    annotation = {'segmentation': polyMag, 
                                  'area': int(template['mSize']), 
                                  'iscrowd':0, 
                                  'image_id':int(adjustedIdPatch), 
                                  'bbox':bboxMag, 
                                  'category_id': 2, 
                                  'id': int(annotationId)}
                    annotationId += 1
                    data["annotations"].append(annotation)

                    log.debug("to add an image to a background, the background is first masked with the invert of the local envelope before adding the image")
                    eMask = template['e']['mask']
                    eMaskInvert = cv.bitwise_not(eMask)
                    
                    bbox = template['bbox']
                    imPatchBox = imPatch[y:y + bbox[3], x:x + bbox[2]]
                    imPatchBox = cv.bitwise_and(imPatchBox, imPatchBox, mask = eMaskInvert)
                    imPatch[y:y+bbox[3], x:x+bbox[2]] = cv.add(imPatchBox, template['e']['im'])

                    fluoPatchBox = fluoPatch[y:y + bbox[3], x:x + bbox[2]]
                    fluoPatchBox = cv.bitwise_and(fluoPatchBox, fluoPatchBox, mask = eMaskInvert)
                    fluoPatch[y:y + bbox[3], x:x + bbox[2]] = cv.add(fluoPatchBox, template['e']['fluo'])

                mergedPatch = np.zeros((self.patch_size[1], self.patch_size[0], 3))
                mergedPatch[:, :, 0] = imPatch
                mergedPatch[:, :, 1] = fluoPatch
                mergedName = f"{str(adjustedIdPatch).zfill(5)}.jpg"
                mergedPath = os.path.join(imageFolder, mergedName)
                cv.imwrite(mergedPath, mergedPatch)

                image_info = {'file_name': os.path.join('images', mergedName), 
                              'id': int(adjustedIdPatch), 
                              'height': int(self.patch_size[0]), 
                              'width': int(self.patch_size[1])}
                data["images"].append(image_info)

            log.info(f"Writing {dataset_type} annotations to: {annotationsPath}")
            with open(annotationsPath, 'w') as f:
                json.dump(data, f)

    def keep_artificial_data_in_wafer_dir(self, wafer_dir):
        """ Copy (keep) artificial data in user's directory

        Since the artificial folder will be cleaned after every training, we provided the option for a user
        to save the artificial data (train and val) in the corresponding wafer directory. 

        Parameters
        ----------
        wafer_dir: str
            Path to the directory where the artificial data will be stored.
        """
        log.info(f"Keep artificial data in {wafer_dir}")
        copy_tree(self.artificial_folder, wafer_dir)

    def delete_train_and_test_data(self, directory=None):
        """ Delete artificial data

        After every pipeline run, we remove training and validation data that is generated in artificial
        patch generator.

        Parameters
        ----------
        directory: str
            Directory that will be removed. If None, it will remove default artificial folder directory.
        """
        log.info(f"Removig artificial data from {directory}")
        directory = directory or self.artificial_folder
        if os.path.exists(directory):
            rmtree(directory)


