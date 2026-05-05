"""
MagReorderer companion of MagFinder

Notes:
    key indexing
        the indexing of sections can have gaps: keys = [0,1,3,5,6]
    continuous indexing
        it is easier to work with continuous indexing
        for ordering computations [0,1,3,5,6] -> [0,1,2,3,4]
    key       -> continuous happens when extracting high res rois: roi_image_filename(id_enumerate),
    continous -> key        happens when assigning the wafer.serial_order: self.wafer.serial_order = [sorted_section_keys[o] for o in order]
"""

import itertools
import os

from fiji.util.gui import GenericDialogPlus
from ij import IJ
from java.awt import Polygon
from java.lang import Math
from java.util.concurrent.atomic import AtomicInteger
from loci.formats import ImageReader, MetadataTools
from net.imglib2.img.display.imagej import ImageJFunctions as IL
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory
from net.imglib2.realtransform import AffineTransform2D
from net.imglib2.realtransform import RealViews as RV
from net.imglib2.view import Views

from src.geometry.transforms import TRANSFORM_SUBPIXEL, TRANSFORM_SUBPIXEL_INVERSE
from src.geometry.geometry import (
    change_basis,
    get_imglib2_transform_scaling,
    to_imglib2_aff,
    transform_points,
    transform_points_to_poly,
)
from src.order.metric import Metric
from src.order.sift import (
    create_sift_parameters,
    deserialize_matchings,
    get_SIFT_similarity_parallel,
    parallel_compute_sift,
    serialize_matchings,
    sift_params_to_string,
)
from src.order.siftmode import SIFTMode
from src.order.tspsolver import compute_tsp_order
from src.utils.combinatorics import pairwise
from src.utils.file import folder_content, get_image_paths, mkdir_p, roi_image_filename
from src.utils.image import norm_local_contrast, open_subpixel_crop, rotate
from src.utils.log import dlog, log_block
from src.utils.matrix import init_mat
from src.utils.numeric import intr
from src.utils.parallel import start_threads
from src.utils.serialize import deserialize_parallel

TRANSFORM_DOWNSAMPLED = AffineTransform2D()
TRANSFORM_DOWNSAMPLED.translate([-0.5, -0.5])
TRANSFORM_DOWNSAMPLED_INVERSE = TRANSFORM_DOWNSAMPLED.inverse()

MAX_DISSIMILARITY = 50000
REORDER_ROI_ID = 0


def get_downsampling_factor(wafer):
    """Downsampling factor between the high-res and low-res images."""
    image_paths = get_image_paths(wafer.root)
    lowres_path = image_paths[0]
    highres_path = image_paths[-1]
    reader = ImageReader()
    ome_metadata = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(ome_metadata)
    try:
        reader.setId(highres_path)
        large_x = ome_metadata.getPixelsSizeX(0).getNumberValue()
        reader.setId(lowres_path)
        small_x = ome_metadata.getPixelsSizeX(0).getNumberValue()
    finally:
        reader.close()
    downsampling_factor = large_x / float(small_x)
    dlog(
        "Downsampling factor between low and high res images: {}".format(
            downsampling_factor
        ).center(100, "-")
    )
    return downsampling_factor


def compute_highres_roi_size(wafer, downsampling_factor):
    """Size of the extracted ROI window in the high resolution image."""
    width = height = 0
    for section in wafer.sections.values():
        if not section.rois:
            continue
        points = [point for roi in section.rois.values() for point in roi.points]
        box = Polygon(
            [int(point[0]) for point in points],
            [int(point[1]) for point in points],
            len(points),
        ).getBounds()
        width = max(width, box.width)
        height = max(height, box.height)
    return intr(Math.sqrt(width * height) * downsampling_factor * 1.5)


def export_highres(
    wafer,
    annotation_types,
    image_path,
    downsampling_factor,
    highres_roi_size,
    channel,
    export_folder,
):
    """Exports high-resolution annotated section images.

    # TODO subpixel issue to revisit with new understanding of its root cause
    """
    dlog("High resolution export ...")
    dir_export = mkdir_p(export_folder)
    half_size = int(0.6 * highres_roi_size)

    scale_upsampling = AffineTransform2D()
    scale_upsampling.scale(downsampling_factor)

    translate_half_window = AffineTransform2D()
    translate_half_window.translate(half_size, half_size)

    for id_serial, id_section in enumerate(wafer.serial_order):
        title = "section_serial_{:04}_magc_{:04}".format(id_serial, id_section)
        section = wafer.sections[id_section]
        centroid_highres = [downsampling_factor * coord for coord in section.centroid]
        im = open_subpixel_crop(
            image_path,
            centroid_highres[0] - half_size,
            centroid_highres[1] - half_size,
            2 * half_size,
            2 * half_size,
            channel,
        )
        rotate(im, section.angle)

        annotation_transform = AffineTransform2D()
        annotation_transform.preConcatenate(section.transform)
        annotation_transform.preConcatenate(scale_upsampling)
        annotation_transform.preConcatenate(translate_half_window)

        # seems to work also in downsampling context
        annotation_transform = change_basis(
            annotation_transform,
            B=TRANSFORM_DOWNSAMPLED,
            B_inverse=TRANSFORM_DOWNSAMPLED_INVERSE,
        )

        for annotation in section.all_annotations():
            if not isinstance(annotation, annotation_types):
                continue
            local_poly = transform_points_to_poly(
                annotation.points, annotation_transform
            )
            local_poly.setName(str(annotation))
            local_poly.setStrokeColor(annotation.color)
            local_poly.setImage(im)
            local_poly.setHandleSize(annotation.handle_size_local)
            im.setRoi(local_poly)
            im = im.flatten()
        im.setTitle(title)
        IJ.save(im, os.path.join(dir_export, "{}.tif".format(title)))
    dlog("High resolution export completed")


class MagReorderer(object):
    """Reordering of sections"""

    def __init__(self, wafer):
        dlog("Starting MagReorderer ...")
        self.user_params = self.get_user_params()
        if self.user_params is None:
            return
        self.wafer = wafer
        self.wafer.manager_to_wafer()  # to compute transforms
        self.n_sections = len(self.wafer)

        self.image_path = self.get_highres_image_path()
        self.downsampling_factor = get_downsampling_factor(self.wafer)
        self.working_folder = mkdir_p(
            os.path.join(self.wafer.root, "ordering_working_folder")
        )
        self.roi_folder = mkdir_p(os.path.join(self.working_folder, "roi_images"))
        self.coarse_features_folder = self.get_features_folder(SIFTMode.COARSE)
        self.fine_features_folder = self.get_features_folder(SIFTMode.FINE)
        self.coarse_sift_matches = mkdir_p(
            os.path.join(self.working_folder, "coarse_matches")
        )
        self.neighbor_fine_sift_matches = mkdir_p(
            os.path.join(self.working_folder, "fine_matches")
        )
        self.sift_order_path = os.path.join(self.working_folder, "sift_order.txt")
        self.highres_roi_size = compute_highres_roi_size(
            self.wafer, self.downsampling_factor
        )

    def get_user_params(self):
        gd = GenericDialogPlus("MagReorderer parameters")
        gd.addNumericField(
            "Percentage of available computer cores for parallel computations",
            90,
            0,
            5,
            "%",
        )
        gd.addMessage("-" * 150)
        gd.addMessage(
            "Is the image used for reordering a multichannel image?"
            '\nIf yes, then tick the "multichannel" box and give the channel number (0-based)'
        )
        gd.addCheckbox("multichannel", False)
        gd.addNumericField("channel", 0, 0)
        gd.addMessage("-" * 150)
        gd.addChoice(
            "Type of transform for fits across sections", ["rigid", "affine"], "rigid"
        )
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
        gd.addNumericField(
            "window size for normalize local contrast ", 100, 0, 5, "pixel"
        )
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
            "\nWith neighborhood=10, each section is compared against its 10 most similar neighbors."
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
        return dict(
            percentage_cores=int(gd.getNextNumber()),
            multichannel=gd.getNextBoolean(),
            channel=int(gd.getNextNumber()),
            transform=gd.getNextChoice(),
            contrast=gd.getNextBoolean(),
            contrast_size=int(gd.getNextNumber()),
            sift_gaussian_1=float(gd.getNextNumber()),
            sift_steps_1=int(gd.getNextNumber()),
            sift_min_octave_1=int(gd.getNextNumber()),
            sift_max_octave_1=int(gd.getNextNumber()),
            neighborhood=int(gd.getNextNumber()),
            sift_gaussian_2=float(gd.getNextNumber()),
            sift_steps_2=int(gd.getNextNumber()),
            sift_min_octave_2=int(gd.getNextNumber()),
            sift_max_octave_2=int(gd.getNextNumber()),
            export_highres=gd.getNextBoolean(),
        )

    def get_lowres_image_path(self):
        return get_image_paths(self.wafer.root)[0]

    def get_highres_image_path(self):
        return get_image_paths(self.wafer.root)[-1]

    def get_sift_parameters(self, sift_mode):
        if sift_mode is SIFTMode.COARSE:
            fields = (
                "sift_gaussian_1",
                "sift_steps_1",
                "sift_min_octave_1",
                "sift_max_octave_1",
            )
        elif sift_mode is SIFTMode.FINE:
            fields = (
                "sift_gaussian_2",
                "sift_steps_2",
                "sift_min_octave_2",
                "sift_max_octave_2",
            )
        return create_sift_parameters(4, *[self.user_params[field] for field in fields])

    def get_features_folder(self, sift_mode):
        """Creates and returns the folder with a string built from the sift parameters"""
        name = sift_params_to_string(self.get_sift_parameters(sift_mode))
        return mkdir_p(os.path.join(self.working_folder, name))

    def reorder(self, annotation_types):
        """
        Reorders the sections based on the ROI defined in each section
        """
        with log_block("extract high res rois"):
            self.extract_high_res_rois()
        with log_block("coarse sift matching"):
            self.get_matches(SIFTMode.COARSE, self.coarse_sift_matches)
        with log_block("compute neighbor pairs"):
            neighbor_pairs = self.compute_neighbor_pairs()
        dlog("neighbor pairs after coarse matching: {}".format(neighbor_pairs))
        with log_block("fine sift matching"):
            self.get_matches(
                SIFTMode.FINE, self.neighbor_fine_sift_matches, pairs=neighbor_pairs
            )
        with log_block("compute order"):
            order_computed = self.compute_order(self.neighbor_fine_sift_matches)
        if not order_computed:
            dlog("Skipping align and export: no order was computed.")
            self.show_roi_stack()
            return
        with log_block("align sections"):
            self.align_sections()
        if self.user_params["export_highres"]:
            with log_block("export highres"):
                export_highres(
                    self.wafer,
                    annotation_types,
                    self.image_path,
                    self.downsampling_factor,
                    self.highres_roi_size,
                    self.user_params["channel"],
                    os.path.join(self.working_folder, "export_high_res"),
                )
        self.show_roi_stack()

    def extract_high_res_rois(self):
        """Extracts the ROIs in the high resolution image.

        We use ROI #0 in every section.
        Note: parallelization has been attempted but failed.
        This sequential version is the current working implementation.
        """
        dlog("Extracting ROI images in {} sections...".format(self.n_sections))
        if len(os.listdir(self.roi_folder)) == self.n_sections:
            dlog("ROI images already extracted")
            return
        channel = (
            self.user_params["channel"] if self.user_params["multichannel"] else None
        )
        for id_enumerate, key in enumerate(sorted(self.wafer.sections)):
            roi = self.wafer.sections[key].rois[REORDER_ROI_ID]
            highres_centroid = [
                roi.centroid[0] * float(self.downsampling_factor),
                roi.centroid[1] * float(self.downsampling_factor),
            ]
            roi_path = os.path.join(self.roi_folder, roi_image_filename(id_enumerate))
            highres_roi_im = open_subpixel_crop(
                self.image_path,
                highres_centroid[0] - 0.5 * self.highres_roi_size,
                highres_centroid[1] - 0.5 * self.highres_roi_size,
                self.highres_roi_size,
                self.highres_roi_size,
                channel,
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
            IJ.save(highres_roi_im, roi_path)

    def compute_features(self, sift_params, features_folder):
        """Computes in parallel sift features of the extracted ROI images"""
        if len(os.listdir(features_folder)) == self.n_sections:
            dlog("Features already computed.")
            return
        roi_paths = folder_content(self.roi_folder)
        dlog("Compute features with sift params {} ...".format(sift_params))
        start_threads(
            parallel_compute_sift,
            fraction_cores=0.01 * self.user_params["percentage_cores"],
            arguments=(AtomicInteger(0), roi_paths, features_folder, sift_params),
        )

    def load_features(self, features_folder):
        """Loads serialized SIFT features from disk."""
        features_paths = folder_content(features_folder)
        all_features = [0] * len(features_paths)
        with log_block("load features"):
            start_threads(
                deserialize_parallel,
                fraction_cores=0.01 * self.user_params["percentage_cores"],
                arguments=[AtomicInteger(0), features_paths, all_features],
            )
        return all_features

    def compute_matches(self, features, matches_folder, pairs=None):
        """Computes in parallel the sift matches among the given pairs."""
        costs = {
            metric: init_mat(self.n_sections, init_value=MAX_DISSIMILARITY)
            for metric in Metric.all()
        }
        affine_transforms = {}

        if pairs is None:
            dlog("Computing all pairwise matches...")
            pairs = list(itertools.combinations(range(self.n_sections), 2))

        translation_threshold = 0.2 * self.highres_roi_size
        dlog("translation threshold {}".format(translation_threshold))
        with log_block("compute SIFT matches"):
            start_threads(
                get_SIFT_similarity_parallel,
                fraction_cores=0.01 * self.user_params["percentage_cores"],
                arguments=(
                    AtomicInteger(0),
                    pairs,
                    features,
                    costs,
                    affine_transforms,
                    self.user_params["transform"],
                    translation_threshold,
                    self.highres_roi_size,
                ),
            )
        serialize_matchings(
            costs,
            affine_transforms,
            matches_folder,
            self.user_params["percentage_cores"],
        )

    def compute_neighbor_pairs(self):
        """Finds the neighbor pairs across all metrics from the coarse matches."""
        return list(
            {
                pair
                for metric in Metric.all()
                for pair in self.get_neighbor_pairs(
                    self.coarse_sift_matches,
                    metric,
                    neighborhood=self.user_params["neighborhood"],
                )
            }
        )

    def get_matches(self, sift_mode, matches_folder, pairs=None):
        """Computes features then matches, skipping each step if already cached."""
        features_folder = self.get_features_folder(sift_mode)

        with log_block("compute features"):
            self.compute_features(self.get_sift_parameters(sift_mode), features_folder)

        if len(os.listdir(matches_folder)) == self.n_sections:
            dlog("Matches already computed.")
            return

        features = self.load_features(features_folder)
        with log_block("compute matches"):
            self.compute_matches(features, matches_folder, pairs=pairs)

    @staticmethod
    def get_cost_mat(
        matches_folder, metric=Metric.INLIER_NUMBER, percentage_cores=0.95
    ):
        """
        Returns a cost matrix using the given metric
        inliers: inverse of the number of remaining inliers after ransac
        displacement: average displacement of the transformed inliers
        """
        (pairwise_costs, _) = deserialize_matchings(matches_folder, percentage_cores)
        return pairwise_costs[metric]

    def get_neighbor_pairs(self, matches_folder, metric, neighborhood=10):
        """Returns the best neighbor pairs given a neighborhood"""
        pairwise_costs = self.get_cost_mat(
            matches_folder,
            metric=metric,
            percentage_cores=self.user_params["percentage_cores"],
        )
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
        """Computes and saves the order given the path of the stored matches.

        Returns True if a valid order was produced or loaded from cache, False otherwise.
        """
        sorted_section_keys = sorted(self.wafer.sections)
        if os.path.isfile(self.sift_order_path):
            dlog("Order already computed. Loading from file ...".center(100, "-"))
            with open(self.sift_order_path, "r") as f:
                self.wafer.serial_order = [
                    sorted_section_keys[int(x)] for x in f.readline().split(",")
                ]
            return True
        pairwise_costs = self.get_cost_mat(
            matches_folder,
            metric=metric,
            percentage_cores=self.user_params["percentage_cores"],
        )
        order = compute_tsp_order(pairwise_costs, self.working_folder)
        if not order:
            dlog("No order computed; leaving serial_order unchanged.")
            return False
        with open(self.sift_order_path, "w") as f:
            f.write(",".join(str(o) for o in order))
        dlog("The order is: {}".format(order))
        self.wafer.serial_order = [sorted_section_keys[o] for o in order]
        return True

    def get_pair_local_transform(self, o1, o2, affine_transforms):
        """Returns the local pairwise transform from section o2 to o1.

        Converts from mpicbg AffineModel2D to imglib2 AffineTransform2D and
        applies subpixel correction. Falls back to identity if the transform
        is missing or has implausible scaling.
        """
        if (o2, o1) not in affine_transforms:
            dlog("Warning: transform missing for pair {}".format((o2, o1)))
            return AffineTransform2D()
        transform = to_imglib2_aff(affine_transforms[(o2, o1)])
        transform = change_basis(
            transform,
            B=TRANSFORM_SUBPIXEL_INVERSE,
            B_inverse=TRANSFORM_SUBPIXEL,
        )
        scaling = get_imglib2_transform_scaling(transform)
        if not (0.9 < scaling < 1.1):
            msg = "Bad scaling {} for pair {}, use identity instead".format(
                scaling, (o1, o2)
            )
            dlog(msg)
            return AffineTransform2D()
        return transform

    def align_sections(self):
        """Realigns the sections based on the transforms found during reordering."""
        affine_transforms = deserialize_matchings(
            self.neighbor_fine_sift_matches, self.user_params["percentage_cores"]
        )[1]

        scale_upsampling = AffineTransform2D()
        scale_upsampling.scale(self.downsampling_factor)
        dlog("downsampling factor {}".format(self.downsampling_factor))

        translation_zero_to_half_highres_fov = AffineTransform2D()
        translation_zero_to_half_highres_fov.translate(
            2 * [0.5 * self.highres_roi_size]
        )

        k1 = self.wafer.serial_order[0]

        # translation global ROI to (0,0)
        translation_ROI_to_zero = AffineTransform2D()
        translation_ROI_to_zero.translate(
            [
                -self.wafer.sections[k1].rois[REORDER_ROI_ID].centroid[0],
                -self.wafer.sections[k1].rois[REORDER_ROI_ID].centroid[1],
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

        ref_local_highres_section = transform_points(
            self.wafer.sections[k1].points,
            ROI_global_lowres_to_local_highres,
        )
        ref_local_highres_ROI = transform_points(
            self.wafer.sections[k1].rois[REORDER_ROI_ID].points,
            ROI_global_lowres_to_local_highres,
        )

        # cumulative_local_transform
        # 1. it is updated as we go from pair to pair
        # 2. it transforms the local low-res image of a section
        # into the local low-res image of the first reference section (in serial order)
        # 3. it is a concatenation of the consecutive local pairwise transforms
        cumulative_local_transform = AffineTransform2D()

        sorted_section_keys = sorted(self.wafer.sections)
        section_key_to_index = {key: idx for idx, key in enumerate(sorted_section_keys)}
        for k1, k2 in pairwise(self.wafer.serial_order):
            o1 = section_key_to_index[k1]
            o2 = section_key_to_index[k2]
            pair_local_transform = self.get_pair_local_transform(
                o1, o2, affine_transforms
            )
            cumulative_local_transform.preConcatenate(pair_local_transform)

            translation_ROI_to_zero = AffineTransform2D()
            translation_ROI_to_zero.translate(
                [
                    -self.wafer.sections[k2].rois[REORDER_ROI_ID].centroid[0],
                    -self.wafer.sections[k2].rois[REORDER_ROI_ID].centroid[1],
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
        self.wafer.compute_transforms()
        self.wafer.wafer_to_manager()
        dlog("The sections have been updated".center(100, "-"))

    def show_roi_stack(self, straight=False):
        half_view = int(0.75 * self.highres_roi_size)
        center = self.highres_roi_size // 2
        view_specs = [
            [center - half_view, center - half_view],
            [center + half_view, center + half_view],
        ]
        (_, affine_transforms) = deserialize_matchings(
            self.neighbor_fine_sift_matches, self.user_params["percentage_cores"]
        )
        if straight:
            affine_transforms = {key: AffineTransform2D() for key in affine_transforms}
        else:
            affine_transforms = {
                key: to_imglib2_aff(transform)
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


def ordered_transformed_imgstack(
    order, affine_transforms, loaded_imgs, view_specs, sorted_section_keys
):
    imgs = [Views.interval(loaded_imgs[order[0]], *view_specs)]
    transform = AffineTransform2D()
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

        dlog("j {}".format(str(j).zfill(4)))
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
