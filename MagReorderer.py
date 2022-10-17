"""
MagReorderer companion of MagFinder
"""
import importlib
import io.scif.img.ImgOpener
import itertools
import os
import threading
import time
from collections import namedtuple

import ij
import java
import loci
import loci.plugins.BF
import net.imagej.ImageJ
from fiji.util.gui import GenericDialogPlus
from ij import IJ, ImagePlus, ImageStack, WindowManager
from ij.gui import GenericDialog, PolygonRoi, Roi
from java.awt.event import KeyAdapter, KeyEvent
from java.io import (
    FileInputStream,
    FileOutputStream,
    ObjectInputStream,
    ObjectOutputStream,
)
from java.lang import Exception as java_exception
from java.lang import Math, Runtime
from java.util import ArrayList, HashSet
from java.util.concurrent.atomic import AtomicInteger
from loci.common import Region
from loci.formats import ImageReader, MetadataTools
from mpicbg.ij import SIFT, FeatureTransform
from mpicbg.ij.plugin import NormalizeLocalContrast
from mpicbg.imagefeatures import FloatArray2DSIFT
from mpicbg.models import AffineModel2D, NotEnoughDataPointsException, PointMatch
from net.imglib2.converter import RealUnsignedByteConverter
from net.imglib2.img.display.imagej import ImageJFunctions as IL
from net.imglib2.img.display.imagej import ImageJVirtualStackUnsignedByte
from net.imglib2.interpolation.randomaccess import (
    NearestNeighborInterpolatorFactory,
    NLinearInterpolatorFactory,
)
from net.imglib2.realtransform import AffineTransform2D
from net.imglib2.realtransform import RealViews as RV
from net.imglib2.view import Views

# from loci.plugins.in import ImporterOptions # fails because of the "in", use import_module instead
ImporterOptions = importlib.import_module("loci.plugins.in.ImporterOptions")

ACCEPTED_IMAGE_FORMATS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
N_SUBROIS = 10


def scale_model2D(model2d, factor, highres_w):
    """Rescales a model2d transform"""
    scale = AffineModel2D()
    scale.set(factor, 0, 0, factor, 0, 0)

    translation = AffineModel2D()
    translation.set(1, 0, 0, 1, highres_w / 2.0, highres_w / 2.0)

    output = AffineModel2D()
    output.preConcatenate(scale)
    output.preConcatenate(translation)
    output.preConcatenate(model2d)
    output.preConcatenate(translation.createInverse())
    output.preConcatenate(scale.createInverse())
    return output


def folder_content(folder):
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))]


def get_OK(text):
    gd = GenericDialog("User prompt")
    gd.addMessage(text)
    gd.hideCancelButton()
    gd.enableYesNoCancel()
    # focus_on_ok(gd)
    gd.showDialog()
    return gd.wasOKed()


def start_threads(function, fractionCores=1, arguments=None, nThreads=None):
    threads = []
    if nThreads == None:
        threadRange = range(
            max(int(Runtime.getRuntime().availableProcessors() * fractionCores), 1)
        )
    else:
        threadRange = range(nThreads)
    IJ.log("Running in parallel with ThreadRange = " + str(threadRange))
    for p in threadRange:
        if arguments == None:
            thread = threading.Thread(target=function)
        else:
            # IJ.log('These are the arguments ' + str(arguments) + 'III type ' + str(type(arguments)))
            thread = threading.Thread(group=None, target=function, args=arguments)
        threads.append(thread)
        thread.start()
        IJ.log("Thread " + str(p) + " started")
    for idThread, thread in enumerate(threads):
        thread.join()
        IJ.log("Thread " + str(idThread) + "joined")


def serialize(x, path):
    object_output_stream = ObjectOutputStream(FileOutputStream(path))
    object_output_stream.writeObject(x)
    object_output_stream.close()


def deserialize(path):
    object_input_stream = ObjectInputStream(FileInputStream(path))
    x = object_input_stream.readObject()
    object_input_stream.close()
    return x


def deserialize_parallel(atom, paths, all_features):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k < len(paths):
            all_features[k] = deserialize(paths[k])


def mkdir_p(path):
    path = os.path.join(path, "")
    try:
        os.mkdir(path)
        IJ.log("Folder created: " + path)
    except Exception as e:
        if e[0] == 20047:
            pass
        else:
            IJ.log("Exception during folder creation :" + str(e))
    return path


def open_crop(crop_params):
    cropped = subpixel_crop_open(
        crop_params.high_res_path,
        crop_params.crop_x,
        crop_params.crop_y,
        crop_params.crop_w,
        crop_params.crop_h,
        crop_params.channel,
    )
    highres_roi_im = subpixel_crop(
        cropped,
        0.5 * (cropped.getWidth() - crop_params.highres_w),
        0.5 * (cropped.getHeight() - crop_params.highres_w),
        crop_params.highres_w,
        crop_params.highres_w,
    )
    return highres_roi_im


def open_crop_parallel(
    atom,
    crop_info,
):
    """There was a problem with the parallelization of this operation"""
    while atom.get() < len(crop_info):
        k = atom.getAndIncrement()
        if k < len(crop_info):
            IJ.log("Open crop rotate image {}".format(k))
            p = crop_info[k]
            IJ.log("crop_info - " + str(p))
            highres_roi_im = open_crop(p)
            highres_roi_im = normLocalContrast(highres_roi_im, 50, 50, 3, True, True)
            IJ.run(highres_roi_im, "8-bit", "")
            IJ.save(highres_roi_im, p.roi_path)


def create_sift_parameters(
    fdSize,
    initialSigma,
    steps,
    minOctaveSize,
    maxOctaveSize,
):
    p = FloatArray2DSIFT.Param().clone()
    p.fdSize = fdSize
    p.initialSigma = initialSigma
    p.steps = steps
    p.maxOctaveSize = maxOctaveSize
    p.minOctaveSize = minOctaveSize
    return p


def sift_params_to_string(sift_params):
    return "{}_{}_{}_{}_{}_{:.3f}".format(
        sift_params.fdBins,
        sift_params.fdSize,
        sift_params.steps,
        sift_params.minOctaveSize,
        sift_params.maxOctaveSize,
        sift_params.initialSigma,
    )


def parallel_compute_sift(atom, paths, saveFolder, sift_params, roi=None):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k >= len(paths):
            continue
        float_SIFT = FloatArray2DSIFT(sift_params)
        ij_SIFT = SIFT(float_SIFT)
        im = IJ.openImage(paths[k])
        if roi is not None:
            im = crop(im, roi)
        ip = im.getProcessor()
        features = HashSet()
        ij_SIFT.extractFeatures(ip, features)
        IJ.log("Section {}: {} features extracted".format(k, features.size()))
        im.close()
        serialize(features, os.path.join(saveFolder, "features_" + str(k).zfill(4)))
        del features


def get_SIFT_similarity(
    atomicI,
    pairs,
    allFeatures,
    pairwise_costs,
    affine_transforms,
    matchPointsDict,
    minMatchedFeatures,
):
    while atomicI.get() < len(pairs):
        k = atomicI.getAndIncrement()
        if k >= len(pairs):
            continue
        id1, id2 = pairs[k]
        if k % 100 == 0:
            IJ.log("Processing pair {} ".format((id1, id2)))
        candidates = ArrayList()
        FeatureTransform.matchFeatures(
            allFeatures[id1], allFeatures[id2], candidates, 0.92
        )
        inliers = ArrayList()
        # model = RigidModel2D()
        model = AffineModel2D()
        try:
            modelFound = model.filterRansac(
                candidates,  # candidates
                inliers,  # inliers
                1000,  # iterations
                20,  # maxDisplacement
                0.001,  # ratioOfConservedFeatures wafer_39_beads
                # minMatchedFeatures,
            )
        except NotEnoughDataPointsException as e:
            modelFound = False
        if modelFound:
            # mean displacement of the remaining matching features
            distance = PointMatch.meanDistance(inliers)
            IJ.log(
                (
                    "model found in section pair {} - {} : distance {} - {} inliers"
                ).format(id1, id2, distance, len(inliers))
            )
            # # metric of number of inliers
            # pairwise_costs[id1][id2] = pairwise_costs[id2][id1] = (
            #     1 / len(inliers) * 100
            # )

            # metric of average displacement of point matches
            pairwise_costs[id1][id2] = pairwise_costs[id2][id1] = distance * 100

            affine_transforms[(id1, id2)] = model.createInverse()
            affine_transforms[(id2, id1)] = model
            matchPointsDict[(id1, id2)] = inliers


def create_empty_magc():
    magc = {
        "sections": {},
        "rois": {},
        "magnets": {},
        "focus": {},
        "landmarksEM": {},
        "serialorder": [],
        "tsporder": [],
    }
    return magc


def centroid(points):
    return polygonroi_from_points(points).getContourCentroid()


def crop_open(im_path, x, y, w, h, channel):
    options = ImporterOptions()
    options.setColorMode(ImporterOptions.COLOR_MODE_GRAYSCALE)
    options.setCrop(True)
    options.setCropRegion(0, Region(int(x), int(y), int(w), int(h)))
    options.setId(im_path)
    if channel is None:
        imps = loci.plugins.BF.openImagePlus(options)
        return imps[0]
    options.setSplitChannels(True)
    imps = loci.plugins.BF.openImagePlus(options)
    return imps[channel]


def subpixel_crop_open(im_path, x, y, w, h, channel):
    im = crop_open(im_path, int(x), int(y), w + 1, h + 1, channel)
    IJ.run(
        im,
        "Translate...",
        "x={} y={} interpolation=Bilinear".format(int(x) - x, int(y) - y),
    )
    return crop(im, Roi(0, 0, w, h))


def rotate(im, angleDegree):
    IJ.run(
        im,
        "Rotate... ",
        "angle=" + str(angleDegree) + " grid=1 interpolation=Bilinear",
    )


def polygonroi_from_points(points):
    xPoly = [point[0] for point in points]
    yPoly = [point[1] for point in points]
    return PolygonRoi(xPoly, yPoly, PolygonRoi.POLYGON)


def subpixel_crop(im, x, y, w, h):
    IJ.run(
        im,
        "Translate...",
        "x={} y={} interpolation=Bilinear".format(int(x) - x, int(y) - y),
    )
    return crop(im, Roi(int(x), int(y), w, h))


def crop(im, roi):
    ip = im.getProcessor()
    ip.setRoi(roi)
    im = ImagePlus("{}_Cropped".format(im.getTitle()), ip.crop())
    return im


def normLocalContrast(im, x, y, stdev, center, stretch):
    NormalizeLocalContrast().run(im.getProcessor(), x, y, stdev, center, stretch)
    return im


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def stack_from_ims(ims):
    width = ims[0].getWidth()
    height = ims[0].getHeight()

    stack = ImageStack(width, height)  # assemble the ImageStack of the channel
    for im in ims:
        stack.addSlice(im.getProcessor())
    imp = ImagePlus("Title", stack)
    imp.setDimensions(1, len(ims), 1)  # these have to be timeframes for trackmate
    return imp


def virtual_stack(img_stack):
    return ImageJVirtualStackUnsignedByte(img_stack, RealUnsignedByteConverter(0, 255))


class listen_to_key(KeyAdapter):
    def keyPressed(this, event):
        event.consume()
        handle_keypress(event)


def add_key_listener_everywhere(myListener):
    for elem in (
        [
            IJ.getImage().getWindow(),
            IJ.getImage().getWindow().getCanvas(),
        ]
        # ui.getDefaultUI().getConsolePane().getComponent(),]
        # IJ.getInstance]
        + list(WindowManager.getAllNonImageWindows())
    ):
        kls = elem.getKeyListeners()
        map(elem.removeKeyListener, kls)
        elem.addKeyListener(myListener)


# # # sift_coarse_params = create_sift_parameters(
# # #     fdSize=4, initialSigma=1.6, steps=3, minOctaveSize=64, maxOctaveSize=600
# # # )
# # # # sift_coarse_params = create_sift_parameters(
# # # # fdSize=4, initialSigma=1.6, steps=3, minOctaveSize=64, maxOctaveSize=450
# # # # )
# # # sift_fine_params = create_sift_parameters(
# # #     fdSize=4, initialSigma=1.6, steps=4, minOctaveSize=32, maxOctaveSize=1500
# # # )

# # wafer 52
# sift_coarse_params = create_sift_parameters(
#     fdSize=4, initialSigma=1.6, steps=6, minOctaveSize=64, maxOctaveSize=2000
# )
# sift_fine_params = create_sift_parameters(
#     fdSize=4, initialSigma=1.6, steps=12, minOctaveSize=16, maxOctaveSize=3000
# )


def intr(x):
    return int(round(x))


def ids_to_id(ids):
    """
    [34,1] -> 341
    [34] -> 340 for backcompatibilty with single ROI per section
    """
    if len(ids) == 2:
        return ids[0] * N_SUBROIS + ids[1]
    return ids[0] * N_SUBROIS


def id_to_ids(id_):
    """341 -> [34, 1]"""
    return int(id_ / N_SUBROIS), int(id_ % N_SUBROIS)


def roi_id_to_section_id(roi_id):
    return id_to_ids(roi_id)[0]


class MagReorderer(object):
    def __init__(self, wafer):
        self.wafer = wafer
        self.wafer.manager_to_wafer()  # to compute transforms
        self.GC = self.wafer.GC  # GeometryCalculator
        self.image_path = self.get_im_path(-1)
        self.downsampling_factor = self.get_downsampling_factor()
        self.user_params = self.get_user_params()
        if self.user_params is None:
            return

        self.working_folder = mkdir_p(
            os.path.join(self.wafer.root, "ordering_working_folder")
        )
        self.roi_folder = mkdir_p(
            os.path.join(
                self.working_folder,
                "roi_images",
            )
        )
        self.coarse_features_folder = self.get_features_folder("coarse")
        self.fine_features_folder = self.get_features_folder("fine")
        self.n_sections = len(self.wafer)

        self.all_coarse_sift_matches = os.path.join(
            self.working_folder, "all_coarse_sift_matches"
        )
        self.neighbor_fine_sift_matches = os.path.join(
            self.working_folder, "neighbor_fine_sift_matches"
        )
        self.sift_order_path = os.path.join(self.working_folder, "sift_order.txt")
        # size of the extracted roi in the high res image
        self.highres_w = intr(
            Math.sqrt(next(iter(self.wafer.rois.values())).area)
            * self.downsampling_factor
        )

    def get_user_params(self):
        p = {}
        gd = GenericDialogPlus("MagReorderer parameters")
        gd.addMessage(
            "Is the image used for reordering a multichannel image?"
            '\nIf yes, then tick the "multichannel" box and give the channel number (0-based)'
        )
        gd.addCheckbox("multichannel", False)
        gd.addNumericField("channel", 0, 0)
        gd.addMessage("-" * 150)
        gd.addMessage(
            "Use normalize local contrast? "
            "\nProbably not needed for fluorescent beads imagery"
            "\nProbably needed for brightfield imagery of the sections"
            "\nIf unsure, try without and check:"
            "\n    -the extracted rois in the folder."
            "\n    -the number of features and matches found in the log"
        )
        gd.addCheckbox("normalize local contrast", False)
        gd.addNumericField("window size for normalize local contrast ", 100, 0)
        gd.addMessage("-" * 150)
        gd.addMessage(
            "SIFT parameters for the first coarse all-to-all pairwise matching"
            "\nStart with default parameters then check the log to see whether too many/few features were used"
        )
        gd.addNumericField("gaussian blur", 1.6, 1)
        gd.addNumericField("steps per octave", 3, 0)
        gd.addNumericField("minimum octave size", 64, 0)
        gd.addNumericField("maximum octave size", 600, 0)
        gd.addMessage("-" * 150)
        gd.addMessage(
            "SIFT parameters for the final neighborhood matching."
            "\nWith neighborhood=10, each section is compared against its 10 closest neighbors."
        )
        gd.addNumericField("neighborhood size", 10, 0)
        gd.addNumericField("gaussian blur", 1.6, 1)
        gd.addNumericField("steps per octave", 3, 0)
        gd.addNumericField("minimum octave size", 32, 0)
        gd.addNumericField("maximum octave size", 1500, 0)
        gd.showDialog()
        if gd.wasCanceled():
            return
        p["multichannel"] = gd.getNextBoolean()
        p["channel"] = int(gd.getNextNumber())
        p["contrast"] = gd.getNextBoolean()
        p["contrast_size"] = int(gd.getNextNumber())
        p["sift_gaussian_1"] = float(gd.getNextNumber())
        p["sift_steps_1"] = int(gd.getNextNumber())
        p["sift_min_octave_1"] = int(gd.getNextNumber())
        p["sift_max_octave_1"] = int(gd.getNextNumber())
        p["neighborhood"] = int(gd.getNextNumber())
        p["sift_gaussian_2"] = float(gd.getNextNumber())
        p["sift_steps_2"] = int(gd.getNextNumber())
        p["sift_min_octave_2"] = int(gd.getNextNumber())
        p["sift_max_octave_2"] = int(gd.getNextNumber())
        return p

    def get_im_path(self, n):
        """
        Gets the n_th image path, sorted by size.
        0 gives the downsampled image
        -1 gives the largest image
        """
        return sorted(
            [
                os.path.join(self.wafer.root, name)
                for name in os.listdir(self.wafer.root)
                if any([name.endswith(x) for x in ACCEPTED_IMAGE_FORMATS])
                and not "verview" in name
            ],
            key=os.path.getsize,
        )[n]

    def get_sift_parameters(self, sift_mode):
        if sift_mode == "coarse":
            return create_sift_parameters(
                4,
                self.user_params["sift_gaussian_1"],
                self.user_params["sift_steps_1"],
                self.user_params["sift_min_octave_1"],
                self.user_params["sift_max_octave_1"],
            )
        elif sift_mode == "fine":
            return create_sift_parameters(
                4,
                self.user_params["sift_gaussian_2"],
                self.user_params["sift_steps_2"],
                self.user_params["sift_min_octave_2"],
                self.user_params["sift_max_octave_2"],
            )

    def get_features_folder(self, sift_mode):
        """Creates and returns the folder with a string built from the sift parameters"""

        return mkdir_p(
            os.path.join(
                self.working_folder,
                sift_params_to_string(self.get_sift_parameters(sift_mode)),
            )
        )

    def get_downsampling_factor(self):
        """
        Downsampling factor between the high-res image (used for reordering)
        and the low-res image (used for MagFinder navigation)
        """
        reader = ImageReader()
        omeMeta = MetadataTools.createOMEXMLMetadata()
        reader.setMetadataStore(omeMeta)
        reader.setId(self.get_im_path(-1))
        large_x = omeMeta.getPixelsSizeX(0).getNumberValue()

        reader.setId(self.get_im_path(0))
        small_x = omeMeta.getPixelsSizeX(0).getNumberValue()

        downsampling_factor = large_x / float(small_x)
        IJ.log(
            "Downsampling factor between low and high res images: {}".format(
                downsampling_factor
            ).center(100, "-")
        )
        return downsampling_factor

    def reorder(self):
        """
        Reorders the sections based on the ROI defined in each section
        """
        # TODO parameters from user
        IJ.log("Reordering ...".center(100, "-"))

        # extract the ROIs in the high-res image
        self.extract_high_res_rois()

        # coarse sift matching
        self.get_matches("coarse", self.all_coarse_sift_matches)

        # determine neighbors based on different metrics
        neighbor_pairs = list(  # must be ordered for parallel processing
            set(  # to remove duplicates
                [
                    pair
                    for metric in ["inliers", "displacement"]
                    for pair in self.get_neighbor_pairs(
                        self.all_coarse_sift_matches,
                        metric,
                        neighborhood=self.user_params["neighborhood"],
                    )
                ]
            )
        )
        IJ.log(
            "These are the neighbor pairs after coarse matching {}".format(
                neighbor_pairs
            )
        )

        # fine sift matching among the neighbor pairs
        self.get_matches(
            "fine",
            self.neighbor_fine_sift_matches,
            pairs=neighbor_pairs,
        )

        # compute order based on neighbor distances
        self.compute_order(self.neighbor_fine_sift_matches)

        # alignment of ordered sections
        self.align_sections()
        # self.show_roi_stack()
        # self.show_straight_roi_stack()

    def extract_high_res_rois(self):
        """Extracts the ROIs in the high res image"""
        IJ.log("Extracting ROI images in {} sections...".format(self.n_sections))
        if len(os.listdir(self.roi_folder)) == self.n_sections:
            IJ.log("ROI images already extracted")
            return
        CropParam = namedtuple(
            "CropParam",
            [
                "key",
                "roi_path",
                "high_res_path",
                "highres_w",
                "crop_x",
                "crop_y",
                "crop_w",
                "crop_h",
                "channel",
            ],
        )
        crop_params = []
        for key in sorted(self.wafer.sections):
            highres_xy = [
                centroid * float(self.downsampling_factor)
                for centroid in self.wafer.rois[ids_to_id([key, 0])].centroid
            ]
            crop_params.append(
                CropParam(
                    key=key,
                    roi_path=os.path.join(self.roi_folder, "roi_{:04}.tif".format(key)),
                    high_res_path=self.image_path,
                    highres_w=self.highres_w,
                    crop_x=highres_xy[0] - self.highres_w,
                    crop_y=highres_xy[1] - self.highres_w,
                    crop_w=2 * self.highres_w,
                    crop_h=2 * self.highres_w,
                    channel=self.user_params["channel"]
                    if self.user_params["multichannel"]
                    else None,
                )
            )
        # # something failing when using in parallel
        # start_threads(
        # open_crop_parallel,
        # nThreads=1,
        # # fractionCores=1,
        # arguments=(AtomicInteger(0), crop_params,),
        # )
        for crop_param in crop_params:
            highres_roi_im = open_crop(crop_param)
            if self.user_params["contrast"]:
                highres_roi_im = normLocalContrast(
                    highres_roi_im,
                    self.user_params["contrast_size"],
                    self.user_params["contrast_size"],
                    3,
                    True,
                    True,
                )
            IJ.run(highres_roi_im, "8-bit", "")
            IJ.save(highres_roi_im, crop_param.roi_path)

    def compute_features(self, sift_params, features_folder):
        """Computes in parallel sift features of the extracted ROI images"""
        if len(os.listdir(features_folder)) == self.n_sections:
            return
        roi_paths = folder_content(self.roi_folder)
        IJ.log(
            "Computing all features in parallel with sift params {} ...".format(
                sift_params
            )
        )
        start_threads(
            parallel_compute_sift,
            fractionCores=1,
            arguments=(
                AtomicInteger(0),
                roi_paths,
                features_folder,
                sift_params,
            ),
        )
        IJ.log("Computing features done.".center(100, "-"))

    def compute_matches(self, features_folder, matches_path, pairs=None):
        """Computes in parallel the sift matches among the given pairs"""
        if os.path.isfile(matches_path):
            IJ.log(
                "The matches with these parameters have already been computed. Loading from file ..."
            )
            return
        IJ.log("Loading all features from file...")
        features_paths = folder_content(features_folder)
        all_features = [0] * len(features_paths)
        start_threads(
            deserialize_parallel,
            fractionCores=1,
            arguments=[
                AtomicInteger(0),
                features_paths,
                all_features,
            ],
        )
        IJ.log("All features loaded")

        costs = self.wafer.tsp_solver.init_mat(self.n_sections, initValue=50000)
        affine_transforms = {}
        matchPointsDict = {}
        minMatchedFeatures = 20

        if pairs is None:
            IJ.log("Computing all pairwise matches...")
            pairs = list(itertools.combinations(range(self.n_sections), 2))

        # compute matches in parallel
        IJ.log("Computing SIFT matches ...".center(100, "-"))
        start_threads(
            get_SIFT_similarity,
            fractionCores=0.95,
            arguments=[
                AtomicInteger(0),
                pairs,
                all_features,
                costs,
                affine_transforms,
                matchPointsDict,
                minMatchedFeatures,
            ],
        )
        serialize([costs, affine_transforms, matchPointsDict], matches_path)
        IJ.log("SIFT matches computed.".center(100, "-"))

    def get_matches(self, sift_mode, matches_path, pairs=None):
        """
        1.computes features
        2.computes matches
        """
        features_folder = self.get_features_folder(sift_mode)
        self.compute_features(self.get_sift_parameters(sift_mode), features_folder)
        self.compute_matches(features_folder, matches_path, pairs=pairs)

    @staticmethod
    def get_cost_mat(matches_path, metric="inliers"):
        """
        Returns a cost matrix using the given metric
        inliers: inverse of the number of remaining inliers after ransac
        displacement: average displacement of the transformed inliers
        """
        pairwise_costs, affine_transforms, matchPointsDict = deserialize(matches_path)
        for key_pair, match_points in matchPointsDict.iteritems():
            i, j = key_pair
            if metric == "inliers":
                pairwise_costs[i][j] = 1000 / float(len(match_points))
            elif metric == "displacement":
                pairwise_costs[i][j] = 100 * PointMatch.meanDistance(match_points)
            pairwise_costs[j][i] = pairwise_costs[i][j]
        return pairwise_costs

    def get_neighbor_pairs(self, matches_path, metric, neighborhood=10):
        """Returns the best neighbor pairs given a neighborhood"""
        pairwise_costs = self.get_cost_mat(matches_path, metric=metric)
        all_neighbor_pairs = []

        for i in range(self.n_sections):
            neighbor_pairs = sorted(
                [
                    [pairwise_costs[i][j], (min(i, j), max(i, j))]
                    for j in range(self.n_sections)
                    if pairwise_costs[i][j]
                    > 1e-5  # with only 3 inliers gives sometimes a displacement~0
                ]
            )[:neighborhood]
            all_neighbor_pairs.extend(neighbor_pairs)

        IJ.log("Pairs from metric {}: {}".format(metric, all_neighbor_pairs))
        return [x[1] for x in all_neighbor_pairs]

    def compute_order(self, matches_path, metric="inliers"):
        """Computes and saves the order given the path of the stored matches"""
        if os.path.isfile(self.sift_order_path):
            IJ.log("Order already computed. Loading from file ...".center(100, "-"))
            with open(self.sift_order_path, "r") as f:
                self.wafer.serialorder = [int(x) for x in f.readline().split(",")]
            return
        IJ.log("Computing order ...".center(100, "-"))
        pairwise_costs = self.get_cost_mat(matches_path, metric=metric)
        order = self.wafer.tsp_solver.compute_tsp_order(pairwise_costs)
        with open(self.sift_order_path, "w") as f:
            f.write(",".join([str(o) for o in order]))
        IJ.log("The order is: {}".format(order))
        self.wafer.serialorder = order

    def align_sections(self):
        """Realigns the sections based on the transforms found during reordering"""
        pairwise_costs, affine_transforms, matchPointsDict = deserialize(
            self.neighbor_fine_sift_matches
        )
        lowres_w = self.highres_w / self.downsampling_factor
        # TODO instead of defining this new default local_roi
        # take the actual ROI of the first section (transform_poly.inverse global roi?)
        local_roi = [
            [-lowres_w / 2, -lowres_w / 2],
            [lowres_w / 2, -lowres_w / 2],
            [lowres_w / 2, lowres_w / 2],
            [-lowres_w / 2, lowres_w / 2],
        ]

        # concatenation of the pairwise transforms needed
        # to go from one local section to the next
        cumulative_local_transform = AffineTransform2D()

        sorted_keys = sorted(self.wafer.sections)
        k1 = sorted_keys[self.wafer.serialorder[0]]
        roi_transform = AffineTransform2D()  # sends the global roi to the center 0,0
        roi_transform.translate(
            [-v for v in self.wafer.rois[ids_to_id([k1, 0])].centroid]
        )

        # add the first section and its roi
        self.wafer.add_roi(
            self.GC.transform_points_to_poly(
                local_roi,
                roi_transform.inverse(),
            ),
            k1,
        )
        self.wafer.add_section(
            self.GC.transform_points_to_poly(
                local_roi,
                roi_transform.inverse(),
            ),
            k1,
        )

        # build the stack, pair by pair
        for o1, o2 in pairwise(self.wafer.serialorder):
            # use sorted keys if section_1, section_3 are present
            # and not section_2.
            k2 = sorted_keys[o2]

            roi_transform = AffineTransform2D()  # sends the roi to the center 0,0
            roi_transform.translate(
                [-v for v in self.wafer.rois[ids_to_id([k2, 0])].centroid]
            )

            # sift_transform: sends one roi to the next
            if (o2, o1) not in affine_transforms:
                # not good, these two sections are supposed to be consecutive
                # as determined by the section order, but no match (hence no transform)
                # has been found between these two.
                # Use identity transform.
                IJ.log(
                    "Warning: transform missing in the pair of sections {}".format(
                        (o2, o1)
                    )
                )
                scaled_sift_transform = AffineTransform2D()
            else:
                scaled_sift_transform = self.GC.to_imglib2_aff(
                    scale_model2D(
                        affine_transforms[(o2, o1)],
                        self.downsampling_factor,
                        self.highres_w,
                    )
                )
                transform_scaling = self.GC.get_imglib2_transform_scaling(
                    scaled_sift_transform
                )
                if not (0.9 < transform_scaling < 1.1):
                    IJ.log(
                        "Warning: bad scaling of the transform for pair {}. Using identity instead".format(
                            (o1, o2)
                        )
                    )
                    scaled_sift_transform = AffineTransform2D()
            # concatenate cumulative_local_transform
            cumulative_local_transform.preConcatenate(scaled_sift_transform)

            section_transform = AffineTransform2D()
            section_transform.preConcatenate(cumulative_local_transform)
            section_transform.preConcatenate(roi_transform.inverse())
            self.wafer.update_section(
                k2,
                section_transform,
                local_roi,
            )
        self.wafer.clear_transforms()
        self.wafer.compute_transforms()
        self.wafer.wafer_to_manager()
        IJ.log("The sections have been updated".center(100, "-"))

    def show_roi_stack(self):
        user_view_size = 1000
        view_specs = [
            [-user_view_size, -user_view_size],
            [user_view_size, user_view_size],
        ]
        pairwise_costs, affine_transforms, matchPointsDict = deserialize(
            self.neighbor_fine_sift_matches
        )

        affine_transforms = {
            key: self.GC.to_imglib2_aff(transform)
            for key, transform in affine_transforms.iteritems()
        }

        loaded_imgs = [
            Views.extendZero(
                IL.wrap(IJ.openImage(os.path.join(self.roi_folder, im_name)))
            )
            for im_name in sorted(os.listdir(self.roi_folder))
        ]

        img_stack = ordered_transformed_imgstack(
            self.wafer.serialorder, affine_transforms, loaded_imgs, view_specs
        )
        IL.show(img_stack)

    def show_straight_roi_stack(self):
        user_view_size = 1000
        view_specs = [
            [-user_view_size, -user_view_size],
            [user_view_size, user_view_size],
        ]

        pairwise_costs, affine_transforms, matchPointsDict = deserialize(
            self.neighbor_fine_sift_matches
        )

        affine_transforms = {key: AffineTransform2D() for key in affine_transforms}

        loaded_imgs = [
            Views.extendZero(
                IL.wrap(IJ.openImage(os.path.join(self.roi_folder, im_name)))
            )
            for im_name in sorted(os.listdir(self.roi_folder))
        ]

        img_stack = ordered_transformed_imgstack(
            self.wafer.serialorder, affine_transforms, loaded_imgs, view_specs
        )
        IL.show(img_stack)

        # IJ.run('Orthogonal Views')

        # key_listener = listen_to_key()
        # add_key_listener_everywhere(key_listener)


# --------------- #
# Interactive reorderer
def handle_keypress(keyEvent):
    # TODO to revive
    global order
    global view_specs

    im = IJ.getImage()
    s = im.getZ() - 1

    keycode = keyEvent.getKeyCode()

    if keycode == KeyEvent.VK_D:
        IJ.log("D " + str(s).zfill(4))
        if s > 0:
            IJ.log("before " + str(order))
            # order[s-1], order[s] = order[s], order[s-1]
            # new_s = s-1
            # update_stack(new_s+1)
            update_stack(s - 1, s)
            IJ.log("after " + str(order))

    if keycode == KeyEvent.VK_F:
        IJ.log("F " + str(s).zfill(4))
        if s < len(order) - 1:
            IJ.log("before " + str(order))
            # order[s+1], order[s] = order[s], order[s+1]
            # new_s = s+1
            # update_stack(new_s+1)
            update_stack(s + 1, s)
            IJ.log("after " + str(order))

    if keycode in [
        KeyEvent.VK_RIGHT,
        KeyEvent.VK_UP,
        KeyEvent.VK_DOWN,
        KeyEvent.VK_LEFT,
        KeyEvent.VK_C,
    ]:

        is_c = keycode == KeyEvent.VK_C
        is_right = keycode == KeyEvent.VK_RIGHT
        is_left = keycode == KeyEvent.VK_LEFT
        is_up = keycode == KeyEvent.VK_UP
        is_down = keycode == KeyEvent.VK_DOWN

        im = IJ.getImage()

        if is_c:
            view_specs = [[min_x, min_y], [max_x, max_y]]
        else:
            delta_x = is_right * user_view_size - is_left * user_view_size

            delta_y = -is_up * user_view_size + is_down * user_view_size

            view_specs[0][0] += delta_x
            view_specs[1][0] += delta_x

            view_specs[0][1] += delta_y
            view_specs[1][1] += delta_y

        img_stack = ordered_transformed_imgstack(
            order, affine_transforms, loaded_imgs, view_specs
        )
        im.setStack(virtual_stack(img_stack))

    if keycode in [
        KeyEvent.VK_COMMA,
        KeyEvent.VK_A,
    ]:
        IJ.run("Previous Slice [<]")

    if keycode in [
        KeyEvent.VK_PERIOD,
        KeyEvent.VK_S,
    ]:
        IJ.run("Next Slice [>]")


def update_stack(a, b):
    """
    a is the target slice position
    b is the origin slice (may be smaller or larger than a)
    """
    global start
    start = time.time()
    global order
    # global img
    # global im

    im = IJ.getImage()

    # swap the order
    order[a], order[b] = order[b], order[a]

    img_stack = ordered_transformed_imgstack(
        order, affine_transforms, loaded_imgs, view_specs
    )

    new_virtual_stack = ImageJVirtualStackUnsignedByte(
        img_stack, RealUnsignedByteConverter(0, 255)
    )
    im.setStack(new_virtual_stack)
    # im.updateVirtualSlice()

    im.setZ(a + 1)  # +1 because 1-based
    # IJ.run('Orthogonal Views')


def ordered_transformed_imgstack(order, affine_transforms, loaded_imgs, view_specs):
    imgs = [Views.interval(loaded_imgs[order[0]], *view_specs)]
    transform = AffineTransform2D()
    for i, j in pairwise(order):
        if (i, j) in affine_transforms:
            current_transform = affine_transforms[(i, j)].copy()
        else:
            IJ.log("Warning: the pair ({},{}) is not in the transforms".format(i, j))
            current_transform = AffineTransform2D()

        current_transform.preConcatenate(transform)
        transform = current_transform.copy()

        print("j", str(j).zfill(4))
        transformed = RV.transform(
            Views.interpolate(
                loaded_imgs[j],
                NLinearInterpolatorFactory(),
                # NearestNeighborInterpolatorFactory(),
            ),
            current_transform,
        )

        imgs.append(Views.interval(transformed, *view_specs))

    # permute to use 3rd dimension as Z instead of C (natural order is XYCZT)
    # https://forum.image.sc/t/imglib2-force-wrapped-imageplus-rai-dimensions-to-xyczt/56461/2
    return Views.permute(Views.addDimension(Views.stack(imgs), 0, 0), 3, 2)
