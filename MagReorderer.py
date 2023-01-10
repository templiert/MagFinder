"""
MagReorderer companion of MagFinder

Notes: 
    key indexing
        the indexing of sections can have gaps: keys = [0,1,3,5,6]
    continuous indexing
        it is easier to work with continuous indexing
        for ordering computations [0,1,3,5,6] -> [0,1,2,3,4]
    key       -> continuous happens when extracting high res rois: "roi_{:04}.tif".format(id_enumerate),
    continous -> key        happens when assigning the wafer.serial_order: self.wafer.serial_order = [sorted_section_keys[o] for o in order]
"""
import importlib
import io.scif.img.ImgOpener
import itertools
import os
import sys
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
from ij.gui import GenericDialog, PointRoi, PolygonRoi, Roi
from java.awt import Polygon, Rectangle
from java.awt.event import KeyAdapter, KeyEvent
from java.awt.geom import AffineTransform
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
from mpicbg.models import (
    AffineModel2D,
    NotEnoughDataPointsException,
    PointMatch,
    RigidModel2D,
)
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


def get_distance(p_1, p_2):
    """Distance between two points"""
    return Math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)


class Metric(object):
    """Enum for distance metrics"""

    INLIER_NUMBER = "inlier"
    INLIER_DISPLACEMENT = "displacement"

    @classmethod
    def all(cls):
        return cls.INLIER_NUMBER, cls.INLIER_DISPLACEMENT


class SIFTMode(object):
    """Modes for SIFT"""

    FINE = "fine"
    COARSE = "coarse"


def dlog(x):
    """Double log to print and IJ.log"""
    IJ.log(x)
    print(x)


def folder_content(folder):
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))]


def start_threads(function, fraction_cores=1, arguments=None, n_threads=None):
    if n_threads is None:
        n_threads = max(
            int(Runtime.getRuntime().availableProcessors() * fraction_cores), 1
        )
    thread_range = range(n_threads)
    dlog("Running in parallel with ThreadRange = " + str(thread_range))
    threads = []
    for p in thread_range:
        if arguments is None:
            thread = threading.Thread(target=function)
        else:
            thread = threading.Thread(group=None, target=function, args=arguments)
        threads.append(thread)
        thread.start()
        dlog("Thread {} started".format(p))
    for id_thread, thread in enumerate(threads):
        thread.join()
        dlog("Thread {} joined".format(id_thread))


def serialize(x, path):
    object_output_stream = ObjectOutputStream(FileOutputStream(path))
    object_output_stream.writeObject(x)
    object_output_stream.close()


def serialize_parallel(atom, objects, paths):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k < len(paths):
            serialize(objects[k], paths[k])


def deserialize(path):
    object_input_stream = ObjectInputStream(FileInputStream(path))
    x = object_input_stream.readObject()
    object_input_stream.close()
    return x


def deserialize_parallel(atom, paths, objects):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k < len(paths):
            objects[k] = deserialize(paths[k])


def serialize_matching_outputs(
    costs,
    affine_transforms,
    folder,
):
    start = time.clock()
    n_sections = len(costs.values()[0])
    list_to_serialize = [
        [
            [costs[metric][i] for metric in Metric.all()],
            {
                (i, j): affine_transforms[(i, j)]
                for j in range(n_sections)
                if (i, j) in affine_transforms
            },
        ]
        for i in range(n_sections)
    ]
    start_threads(
        serialize_parallel,
        fraction_cores=0.95,
        arguments=(
            AtomicInteger(0),
            list_to_serialize,
            [
                os.path.join(folder, "matches_section_{:04}".format(i))
                for i in range(n_sections)
            ],
        ),
    )
    dlog("Duration serialize matching outputs: " + str(time.clock() - start))


def deserialize_matching_outputs(folder):
    start = time.clock()
    filenames = sorted(os.listdir(folder))
    deserialized_list = [None] * len(filenames)
    start_threads(
        deserialize_parallel,
        fraction_cores=0.95,
        arguments=(
            AtomicInteger(0),
            [os.path.join(folder, filename) for filename in filenames],
            deserialized_list,
        ),
    )
    costs = {metric: [] for metric in Metric.all()}
    affine_transforms = {}
    for (
        section_costs,
        section_affine_transforms,
    ) in deserialized_list:
        for metric, cost in zip(Metric.all(), section_costs):
            costs[metric].append(cost)
        affine_transforms.update(section_affine_transforms)
    dlog("Duration deserialize matching outputs: " + str(time.clock() - start))
    return costs, affine_transforms


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
    return "{}_{}_{}_{}_{}_{:.1f}".format(
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
        dlog("Section {}: {} features extracted".format(k, features.size()))
        im.close()
        serialize(features, os.path.join(saveFolder, "features_{:04}".format(k)))
        del features


def get_SIFT_similarity_parallel(
    atom,
    pairs,
    section_features,
    pairwise_costs,
    affine_transforms,
    transform_type,
    translation_threshold,
    highres_w,
):
    translation_center = AffineTransform()
    translation_center.translate(0.5 * highres_w, 0.5 * highres_w)
    while atom.get() < len(pairs):
        k = atom.getAndIncrement()
        if k >= len(pairs):
            continue
        id1, id2 = pairs[k]
        if k % 100 == 0:
            dlog("Processing pair {} ".format((id1, id2)))
        get_SIFT_similarity(
            id1,
            id2,
            section_features[id1],
            section_features[id2],
            pairwise_costs,
            affine_transforms,
            transform_type,
            translation_threshold,
            highres_w,
            translation_center,
            0,  # attempt
        )


def change_basis(A, B):
    """Change A into B basis"""
    aff = AffineTransform()
    aff.concatenate(B.createInverse())
    aff.concatenate(A)
    aff.concatenate(B)
    return aff


def translation_norm(A):
    """Returns the norm of the translation vector of the affine transorm"""
    return Math.sqrt(A.getTranslateX() ** 2 + A.getTranslateY() ** 2)


def inliers_to_polygons(inliers):
    """For debugging purposes"""
    p1, p2 = Polygon(), Polygon()
    for inlier in inliers:
        p1.addPoint(*[int(a) for a in inlier.getP1().getL()])
        p2.addPoint(*[int(a) for a in inlier.getP2().getL()])
    return p1, p2


def features_to_polygon(features):
    """For debugging purposes"""
    p = Polygon()
    for feature in features:
        p.addPoint(*[int(a) for a in feature.location])
    return p


def filter_features(features, center, radius):
    """Returns features located within center with radius"""
    filtered_features = HashSet()
    for feature in features:
        if get_distance(feature.location, center) < radius:
            filtered_features.add(feature)
    return filtered_features


def get_SIFT_similarity(
    id1,
    id2,
    features_1,
    features_2,
    pairwise_costs,
    affine_transforms,
    transform_type,
    translation_threshold,
    highres_w,
    translation_center,
    attempt,
):
    debug = False
    pair = id1, id2
    dlog("processing pair {}".format(pair))
    highres_center = [0.5 * highres_w] * 2
    candidates = ArrayList()
    FeatureTransform.matchFeatures(features_1, features_2, candidates, 0.92)
    inliers = ArrayList()

    model = RigidModel2D() if transform_type == "rigid" else AffineModel2D()
    try:
        model_found = model.filterRansac(
            candidates,  # candidates
            inliers,  # inliers
            1000,  # iterations
            20,  # maxDisplacement
            0.001,  # ratioOfConservedFeatures wafer_39_beads
        )
    except NotEnoughDataPointsException as e:
        model_found = False
        dlog("no model found {}".format(pair))
    if model_found:
        affine_transforms[(id2, id1)] = model
        affine_transforms[pair] = model.createInverse()
        inlier_displacement = 100 * PointMatch.meanDistance(inliers)
        inlier_number = 1000 / float(len(inliers))
        center_rebased_transform = change_basis(
            model.createAffine(), translation_center
        )
        dlog(
            (
                "model found in section pair {} | distance {:.1f} | {} inliers"
                " | center_rebased_transform {} | translation norm {}"
            ).format(
                pair,
                inlier_displacement,
                len(inliers),
                center_rebased_transform,
                translation_norm(center_rebased_transform),
            )
        )
        if debug:
            p1, p2 = inliers_to_polygons(inliers)
        if over_translation(center_rebased_transform, translation_threshold):
            dlog("Over translation {}".format(pair).center(25, "-"))
            if debug:
                for p, id in zip((p1, p2), pair):
                    im = IJ.openImage(os.path.join(root, "roi_{:04}.tif".format(id)))
                    im.show()
                    im.setRoi(PointRoi(p), True)
                    im.setTitle("{}_{}".format(pair, id))

            filtering_radius = 0.5 * (1 - 0.1 * (attempt + 1)) * highres_w
            dlog("filtering radius {}".format(filtering_radius))
            new_features_1 = filter_features(
                features_1, highres_center, filtering_radius
            )
            new_features_2 = filter_features(
                features_2, highres_center, filtering_radius
            )
            if len(new_features_2) < 5:
                dlog("Less than 5 features after filtering pair {}".format(pair))
                return
            dlog(
                "len features_2 {}, new_features_2 {}".format(
                    len(features_2), len(new_features_2)
                )
            )
            if debug:
                new_features_pointroi = PointRoi(features_to_polygon(new_features_2))
                dlog("A".center(100, "-"))
                im2 = IJ.openImage(os.path.join(root, "roi_{:04}.tif".format(id2)))
                im2.show()
                im2.setRoi(new_features_pointroi, True)
            get_SIFT_similarity(
                id1,
                id2,
                new_features_1,
                new_features_2,
                pairwise_costs,
                affine_transforms,
                transform_type,
                translation_threshold,
                highres_w,
                translation_center,
                attempt + 1,
            )
        else:
            # metric of average displacement of point matches
            pairwise_costs[Metric.INLIER_DISPLACEMENT][id1][id2] = inlier_displacement
            pairwise_costs[Metric.INLIER_DISPLACEMENT][id2][id1] = inlier_displacement
            # metric of number of inliers
            pairwise_costs[Metric.INLIER_NUMBER][id1][id2] = inlier_number
            pairwise_costs[Metric.INLIER_NUMBER][id2][id1] = inlier_number


def expand_rectangle(rectangle, percentage):
    new_width = rectangle.getWidth() * (100 + percentage) / float(100)
    new_height = rectangle.getHeight() * (100 + percentage) / float(100)
    new_x = rectangle.getX() - 0.5 * percentage / float(100) * rectangle.getWidth()
    new_y = rectangle.getY() - 0.5 * percentage / float(100) * rectangle.getHeight()
    new_rectangle = Rectangle(
        int(new_x),
        int(new_y),
        int(new_width),
        int(new_height),
    )
    return new_rectangle


def over_translation(aff, translation_threshold):
    """
    Compares the norm of the translation to a threshold.
    affine_transform is a AffineModel2D (not AffineTransform2D)
    """
    return (
        Math.sqrt(aff.getTranslateX() ** 2 + aff.getTranslateY() ** 2)
        > translation_threshold
    )


def crop_open(im_path, x, y, w, h, channel):
    """
    Opens only the given ROI in an image saved on disk
    Works with .tif images that do not fit into memory
    x,y,w,h are int
    """
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(w, int)
    assert isinstance(h, int)
    options = ImporterOptions()
    options.setColorMode(ImporterOptions.COLOR_MODE_GRAYSCALE)
    options.setCrop(True)
    options.setCropRegion(0, Region(x, y, w, h))
    options.setId(im_path)
    if channel is None:
        imps = loci.plugins.BF.openImagePlus(options)
        return imps[0]
    options.setSplitChannels(True)
    imps = loci.plugins.BF.openImagePlus(options)
    return imps[channel]


def open_subpixel_crop(im_path, x, y, w, h, channel):
    """Opens only the given ROI with subpixel accuracy. See crop_open"""
    # im = crop_open(im_path, int(x), int(y), w + 1, h + 1, channel)
    im = crop_open(im_path, int(x), int(y), w, h, channel)
    IJ.run(
        im,
        "Translate...",
        " ".join(
            (
                "x={}".format(int(x) - x),
                "y={}".format(int(y) - y),
                "interpolation=Bicubic",
            )
        ),
    )
    return im
    # return crop(im, Roi(0, 0, w, h))


def rotate(im, angle_degree):
    IJ.run(
        im,
        "Rotate... ",
        "angle={} grid=1 interpolation=Bilinear".format(angle_degree),
    )


def polygonroi_from_points(points):
    xPoly = [point[0] for point in points]
    yPoly = [point[1] for point in points]
    return PolygonRoi(xPoly, yPoly, PolygonRoi.POLYGON)


def crop(im, roi):
    ip = im.getProcessor()
    ip.setRoi(roi)
    im = ImagePlus("{}_cropped".format(im.getTitle()), ip.crop())
    return im


def norm_local_contrast(im, x, y, stdev, center, stretch):
    NormalizeLocalContrast().run(im.getProcessor(), x, y, stdev, center, stretch)
    return im


def pairwise(iterable):
    """pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
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


def intr(x):
    return int(round(x))


class MagReorderer(object):
    def __init__(self, wafer):
        dlog("Starting MagReorderer ...")
        self.user_params = self.get_user_params()
        if self.user_params is None:
            return
        self.wafer = wafer
        self.wafer.manager_to_wafer()  # to compute transforms
        self.GC = self.wafer.GC  # GeometryCalculator
        self.image_path = self.get_im_path(-1)
        self.downsampling_factor = self.get_downsampling_factor()
        self.working_folder = mkdir_p(
            os.path.join(self.wafer.root, "ordering_working_folder")
        )
        self.roi_folder = mkdir_p(os.path.join(self.working_folder, "roi_images"))
        self.coarse_features_folder = self.get_features_folder(SIFTMode.COARSE)
        self.fine_features_folder = self.get_features_folder(SIFTMode.FINE)
        self.n_sections = len(self.wafer)
        self.all_coarse_sift_matches = mkdir_p(
            os.path.join(self.working_folder, "coarse_matches")
        )
        self.neighbor_fine_sift_matches = mkdir_p(
            os.path.join(self.working_folder, "fine_matches")
        )
        self.sift_order_path = os.path.join(self.working_folder, "sift_order.txt")
        # size of the extracted roi in the high res image
        self.highres_w = intr(
            Math.sqrt(next(iter(self.wafer.rois.values()))[0].area)
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
        gd.addMessage("Type of transform for fits across sections")
        gd.addChoice("", ["rigid", "affine"], "rigid")
        gd.addMessage("-" * 150)
        gd.addMessage(
            "Use normalize local contrast? "
            "\nProbably not needed for fluorescent beads imagery"
            "\nProbably needed for brightfield imagery of the sections"
            "\nIf unsure, try without and check:"
            "\n    -the extracted rois in the folder."
            "\n    -the number of features and matches found in the log"
        )
        gd.addCheckbox("normalize local contrast", True)
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
        gd.addMessage("-" * 150)
        gd.addCheckbox("export high resolution stack", True)
        gd.showDialog()
        if gd.wasCanceled():
            return
        p["multichannel"] = gd.getNextBoolean()
        p["channel"] = int(gd.getNextNumber())
        p["transform"] = gd.getNextChoice()
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
        p["export_highres"] = gd.getNextBoolean()
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
        if sift_mode is SIFTMode.COARSE:
            return create_sift_parameters(
                4,
                self.user_params["sift_gaussian_1"],
                self.user_params["sift_steps_1"],
                self.user_params["sift_min_octave_1"],
                self.user_params["sift_max_octave_1"],
            )
        elif sift_mode is SIFTMode.FINE:
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
        dlog(
            "Downsampling factor between low and high res images: {}".format(
                downsampling_factor
            ).center(100, "-")
        )
        return downsampling_factor

    def reorder(self):
        """
        Reorders the sections based on the ROI defined in each section
        """
        dlog("Reordering ...".center(100, "-"))

        # extract the ROIs in the high-res image
        self.extract_high_res_rois()

        # coarse sift matching
        self.get_matches(SIFTMode.COARSE, self.all_coarse_sift_matches)
        # return

        # determine neighbors based on different metrics
        start = time.clock()
        neighbor_pairs = list(  # must be ordered for parallel processing
            set(  # to remove duplicates
                [
                    pair
                    for metric in Metric.all()
                    for pair in self.get_neighbor_pairs(
                        self.all_coarse_sift_matches,
                        metric,
                        neighborhood=self.user_params["neighborhood"],
                    )
                ]
            )
        )
        dlog("computing neighbor_pairs took " + str(time.clock() - start))
        dlog(
            "These are the neighbor pairs after coarse matching {}".format(
                neighbor_pairs
            )
        )

        # fine sift matching among the neighbor pairs
        start = time.clock()
        self.get_matches(
            SIFTMode.FINE,
            self.neighbor_fine_sift_matches,
            pairs=neighbor_pairs,
        )
        dlog("get_matches took " + str(time.clock() - start))

        # compute order based on neighbor distances
        self.compute_order(self.neighbor_fine_sift_matches)

        # alignment of ordered sections
        self.align_sections()
        if self.user_params["export_highres"]:
            self.export_highres()
        # self.show_roi_stack()
        # self.show_straight_roi_stack()

    def extract_high_res_rois(self):
        """Extracts the ROIs in the high res image"""
        dlog("Extracting ROI images in {} sections...".format(self.n_sections))
        if len(os.listdir(self.roi_folder)) == self.n_sections:
            dlog("ROI images already extracted")
            return
        CropParam = namedtuple(
            "CropParam",
            [
                "roi_path",
                "highres_path",
                "highres_w",
                "highres_centroid",
                "channel",
            ],
        )
        crop_params = []
        for id_enumerate, key in enumerate(sorted(self.wafer.sections)):
            roi = self.wafer.rois[key][0]
            highres_centroid = [
                roi.centroid[0] * float(self.downsampling_factor),
                roi.centroid[1] * float(self.downsampling_factor),
            ]
            crop_params.append(
                CropParam(
                    roi_path=os.path.join(
                        self.roi_folder,
                        "roi_{:04}.tif".format(id_enumerate),
                    ),
                    highres_path=self.image_path,
                    highres_w=self.highres_w,
                    highres_centroid=highres_centroid,
                    channel=self.user_params["channel"]
                    if self.user_params["multichannel"]
                    else None,
                )
            )
        # # TODO something failing when using in parallel
        # start_threads(
        # open_crop_parallel,
        # n_threads=1,
        # # fraction_cores=1,
        # arguments=(AtomicInteger(0), crop_params,),
        # )
        for crop_param in crop_params:
            highres_roi_im = open_subpixel_crop(
                crop_param.highres_path,
                crop_param.highres_centroid[0] - 0.5 * crop_param.highres_w,
                crop_param.highres_centroid[1] - 0.5 * crop_param.highres_w,
                crop_param.highres_w,
                crop_param.highres_w,
                crop_param.channel,
            )
            if self.user_params["contrast"]:
                highres_roi_im = norm_local_contrast(
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
            dlog("Features already computed.")
            return
        roi_paths = folder_content(self.roi_folder)
        dlog(
            "Compute all features in parallel with sift params {} ...".format(
                sift_params
            )
        )
        start_threads(
            parallel_compute_sift,
            fraction_cores=1,
            arguments=(
                AtomicInteger(0),
                roi_paths,
                features_folder,
                sift_params,
            ),
        )
        dlog("Compute features done.".center(100, "-"))

    def compute_matches(self, features_folder, matches_folder, pairs=None):
        """Computes in parallel the sift matches among the given pairs"""
        if len(os.listdir(matches_folder)) == self.n_sections:
            dlog(
                "The matches with these parameters have already been computed. Loading from file ..."
            )
            return
        dlog("Loading all features from file...")
        features_paths = folder_content(features_folder)
        all_features = [0] * len(features_paths)
        start_threads(
            deserialize_parallel,
            fraction_cores=1,
            arguments=[
                AtomicInteger(0),
                features_paths,
                all_features,
            ],
        )
        dlog("All features loaded")

        costs = {
            metric: self.wafer.tsp_solver.init_mat(self.n_sections, initValue=50000)
            for metric in Metric.all()
        }
        affine_transforms = {}

        if pairs is None:
            dlog("Computing all pairwise matches...")
            pairs = list(itertools.combinations(range(self.n_sections), 2))

        translation_threshold = 0.2 * self.highres_w
        dlog("translation threshold {}".format(translation_threshold))
        # compute matches in parallel
        dlog("Compute SIFT matches ...".center(100, "-"))
        start_threads(
            get_SIFT_similarity_parallel,
            fraction_cores=0.95,
            arguments=[
                AtomicInteger(0),
                pairs,
                all_features,
                costs,
                affine_transforms,
                self.user_params["transform"],
                translation_threshold,
                self.highres_w,
            ],
        )
        serialize_matching_outputs(
            costs,
            affine_transforms,
            matches_folder,
        )
        dlog("SIFT matches computed.".center(100, "-"))

    def get_matches(self, sift_mode, matches_folder, pairs=None):
        """
        1.computes features
        2.computes matches
        """
        features_folder = self.get_features_folder(sift_mode)
        self.compute_features(self.get_sift_parameters(sift_mode), features_folder)
        self.compute_matches(features_folder, matches_folder, pairs=pairs)

    @staticmethod
    def get_cost_mat(matches_folder, metric=Metric.INLIER_NUMBER):
        """
        Returns a cost matrix using the given metric
        inliers: inverse of the number of remaining inliers after ransac
        displacement: average displacement of the transformed inliers
        """
        (
            pairwise_costs,
            _,
        ) = deserialize_matching_outputs(matches_folder)
        return pairwise_costs[metric]

    def get_neighbor_pairs(self, matches_folder, metric, neighborhood=10):
        """Returns the best neighbor pairs given a neighborhood"""
        pairwise_costs = self.get_cost_mat(matches_folder, metric=metric)
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

        dlog("Pairs from metric {}: {}".format(metric, all_neighbor_pairs))
        return [x[1] for x in all_neighbor_pairs]

    def compute_order(self, matches_folder, metric=Metric.INLIER_NUMBER):
        """Computes and saves the order given the path of the stored matches"""
        sorted_section_keys = sorted(self.wafer.sections)
        if os.path.isfile(self.sift_order_path):
            dlog("Order already computed. Loading from file ...".center(100, "-"))
            with open(self.sift_order_path, "r") as f:
                self.wafer.serial_order = [
                    sorted_section_keys[int(x)] for x in f.readline().split(",")
                ]
            return
        dlog("Computing order ...".center(100, "-"))
        pairwise_costs = self.get_cost_mat(matches_folder, metric=metric)
        order = self.wafer.tsp_solver.compute_tsp_order(pairwise_costs)
        with open(self.sift_order_path, "w") as f:
            f.write(",".join(str(o) for o in order))
        dlog("The order is: {}".format(order))
        self.wafer.serial_order = [sorted_section_keys[o] for o in order]

    def align_sections(self):
        """Realigns the sections based on the transforms found during reordering"""
        affine_transforms = deserialize_matching_outputs(
            self.neighbor_fine_sift_matches
        )[1]

        scale_upsampling = AffineTransform2D()
        scale_upsampling.scale(self.downsampling_factor)

        translation_zero_to_half_highres_fov = AffineTransform2D()
        translation_zero_to_half_highres_fov.translate(2 * [0.5 * self.highres_w])

        k1 = self.wafer.serial_order[0]

        # translation global ROI to (0,0)
        translation_ROI_to_zero = AffineTransform2D()
        translation_ROI_to_zero.translate(
            [
                -self.wafer.rois[k1][0].centroid[0],
                -self.wafer.rois[k1][0].centroid[1],
            ]
        )

        # ROI global lowres -> local highres
        # 1. translate to center
        # 2. scale up
        # 3. translate half high res window
        ROI_global_lowres_to_local_highres = AffineTransform2D()
        ROI_global_lowres_to_local_highres.preConcatenate(translation_ROI_to_zero)
        ROI_global_lowres_to_local_highres.preConcatenate(scale_upsampling)
        ROI_global_lowres_to_local_highres.preConcatenate(
            translation_zero_to_half_highres_fov
        )

        ref_local_highres_section = self.GC.transform_points(
            self.wafer.sections[k1].points,
            ROI_global_lowres_to_local_highres,
        )
        ref_local_highres_ROI = self.GC.transform_points(
            self.wafer.rois[k1][0].points,
            ROI_global_lowres_to_local_highres,
        )

        # cumulative_local_transform
        # 1. it is updated as we go from pair to pair
        # 2. it transforms the local low-res image of a section
        # into the local low-res image of the first reference section (in serial order)
        # 3. it is a concatenation of the consecutive local pairwise transforms
        cumulative_local_transform = AffineTransform2D()

        # build the stack, pair by pair
        sorted_section_keys = sorted(self.wafer.sections)
        for k1, k2 in pairwise(self.wafer.serial_order):
            o1 = sorted_section_keys.index(k1)
            o2 = sorted_section_keys.index(k2)

            # compute pair_local_transform:
            # it transforms the local view image of one ROI
            # to the previous ROI, in serial order, at high resolution
            if (o2, o1) not in affine_transforms:
                # bad case: these two sections are supposed to be consecutive
                # as determined by the section order
                # but no match has been found between their ROIs.
                # Use identity transform instead.
                dlog(
                    "Warning: transform missing for"
                    " this pair of sections {}".format((o2, o1))
                )
                pair_local_transform = AffineTransform2D()
            else:
                pair_local_transform = self.GC.to_imglib2_aff(
                    affine_transforms[(o2, o1)]
                )
                transform_scaling = self.GC.get_imglib2_transform_scaling(
                    pair_local_transform
                )
                if not (0.9 < transform_scaling < 1.1):
                    dlog(
                        "Warning: bad scaling of the transform for pair {}."
                        " Using identity instead".format((o1, o2))
                    )
                    pair_local_transform = AffineTransform2D()
            # concatenate cumulative_local_transform
            cumulative_local_transform.preConcatenate(pair_local_transform)

            translation_ROI_to_zero = AffineTransform2D()
            translation_ROI_to_zero.translate(
                [
                    -self.wafer.rois[k2][0].centroid[0],
                    -self.wafer.rois[k2][0].centroid[1],
                ]
            )
            ROI_global_lowres_to_local_highres = AffineTransform2D()
            ROI_global_lowres_to_local_highres.preConcatenate(
                cumulative_local_transform
            )
            ROI_global_lowres_to_local_highres.preConcatenate(
                translation_zero_to_half_highres_fov.inverse()
            )
            ROI_global_lowres_to_local_highres.preConcatenate(
                scale_upsampling.inverse()
            )
            ROI_global_lowres_to_local_highres.preConcatenate(
                translation_ROI_to_zero.inverse()
            )
            self.wafer.update_section(
                k2,
                ROI_global_lowres_to_local_highres,
                ref_local_highres_section,
                ref_local_highres_ROI,
            )
        self.wafer.clear_transforms()
        self.wafer.compute_transforms()
        self.wafer.wafer_to_manager()
        dlog("The sections have been updated".center(100, "-"))

    def export_highres(self):
        dlog("High res export ...")
        dir_export = mkdir_p(os.path.join(self.working_folder, "export_high_res"))
        halfsize = int(0.6 * self.highres_w)
        high_res_path = self.get_im_path(-1)
        for id_enum, id_section in enumerate(self.wafer.serial_order):
            section = self.wafer.sections[id_section]
            im = open_subpixel_crop(
                high_res_path,
                section.centroid[0] * self.downsampling_factor - halfsize,
                section.centroid[1] * self.downsampling_factor - halfsize,
                2 * halfsize,
                2 * halfsize,
                self.user_params["channel"],
            )
            rotate(im, section.angle)
            IJ.save(
                im,
                os.path.join(
                    dir_export, "section_{:04}_{:04}.tif".format(id_enum, id_section)
                ),
            )
        dlog("High res export completed")

    def show_roi_stack(self):
        user_view_size = 1000
        view_specs = [
            [-user_view_size, -user_view_size],
            [user_view_size, user_view_size],
        ]
        (
            _,
            affine_transforms,
        ) = deserialize_matching_outputs(self.neighbor_fine_sift_matches)

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
            self.wafer.serial_order,
            affine_transforms,
            loaded_imgs,
            view_specs,
            sorted(self.wafer.sections),
        )
        IL.show(img_stack)

    def show_straight_roi_stack(self):
        user_view_size = 1000
        view_specs = [
            [-user_view_size, -user_view_size],
            [user_view_size, user_view_size],
        ]

        (
            _,
            affine_transforms,
        ) = deserialize_matching_outputs(self.neighbor_fine_sift_matches)

        affine_transforms = {key: AffineTransform2D() for key in affine_transforms}

        loaded_imgs = [
            Views.extendZero(
                IL.wrap(IJ.openImage(os.path.join(self.roi_folder, im_name)))
            )
            for im_name in sorted(os.listdir(self.roi_folder))
        ]

        img_stack = ordered_transformed_imgstack(
            self.wafer.serial_order,
            affine_transforms,
            loaded_imgs,
            view_specs,
            sorted(self.wafer.sections),
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
        dlog("D " + str(s).zfill(4))
        if s > 0:
            dlog("before " + str(order))
            # order[s-1], order[s] = order[s], order[s-1]
            # new_s = s-1
            # update_stack(new_s+1)
            update_stack(s - 1, s)
            dlog("after " + str(order))

    if keycode == KeyEvent.VK_F:
        dlog("F " + str(s).zfill(4))
        if s < len(order) - 1:
            dlog("before " + str(order))
            # order[s+1], order[s] = order[s], order[s+1]
            # new_s = s+1
            # update_stack(new_s+1)
            update_stack(s + 1, s)
            dlog("after " + str(order))

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


def ordered_transformed_imgstack(
    order, affine_transforms, loaded_imgs, view_specs, sorted_section_keys
):
    imgs = [Views.interval(loaded_imgs[order[0]], *view_specs)]
    transform = AffineTransform2D()
    # for i, j in pairwise(order):
    for k1, k2 in pairwise(order):
        i = sorted_section_keys.index(k1)
        j = sorted_section_keys.index(k2)
        if (i, j) in affine_transforms:
            current_transform = affine_transforms[(i, j)].copy()
        else:
            dlog("Warning: the pair ({},{}) is not in the transforms".format(i, j))
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
