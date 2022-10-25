from __future__ import with_statement

import copy
import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager

import ConfigParser
import ij
import jarray
import java
from fiji.util.gui import GenericDialogPlus
from ij import IJ, ImagePlus, ImageStack, WindowManager
from ij.gui import GenericDialog, PointRoi, PolygonRoi
from ij.io import DirectoryChooser
from ij.plugin import MontageMaker
from ij.plugin.frame import RoiManager
from Jama import Matrix
from java.awt import Button, Checkbox, Color, Rectangle
from java.awt.event import ActionListener, KeyAdapter, KeyEvent, MouseAdapter
from java.awt.geom.Point2D import Float as pFloat
from java.io import File, FileInputStream
from java.lang import Exception as java_exception
from java.lang import Math, Runtime, System
from java.lang.reflect import Array
from java.net import URL
from java.nio.file import Files, Paths
from java.util import HashSet
from java.util.concurrent.atomic import AtomicInteger
from java.util.zip import GZIPInputStream
from mpicbg.models import Point, PointMatch, RigidModel2D
from net.imglib2.img.display.imagej import ImageJFunctions as IL
from net.imglib2.interpolation.randomaccess import (
    NearestNeighborInterpolatorFactory,
    NLinearInterpolatorFactory,
)
from net.imglib2.realtransform import AffineTransform2D
from net.imglib2.realtransform import RealViews as RV
from net.imglib2.view import Views
from org.apache.commons.io import FileUtils

sys.path.append(IJ.getDirectory("plugins"))

import MagReorderer

SIZE_HANDLE = 15
LOCAL_SIZE_STANDARD = 400  # for local summary
# DISPLAY_FACTOR = 2.5
DISPLAY_FACTOR = 2
MSG_DRAWN_ROI_MISSING = (
    "Please draw something before pressing [a]."
    + "\nAfter closing this message you can press [h] for help."
)
ACCEPTED_IMAGE_FORMATS = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
)


def dlog(x):
    """Double log to print and IJ.log"""
    IJ.log(x)
    print(x)


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


class Mode(object):
    GLOBAL = "global"
    LOCAL = "local"


class AnnotationTypeDef(object):
    """Parameters associated to an annotation type"""

    def __init__(self, name, string, color, handle_size_global, handle_size_local):
        self.name = name
        self.string = string
        self.color = color
        self.handle_size_global = handle_size_global
        self.handle_size_local = handle_size_local


class AnnotationType(object):
    SECTION = AnnotationTypeDef("sections", "section", Color.blue, 15, 15)
    ROI = AnnotationTypeDef("rois", "roi", Color.yellow, 15, 15)
    FOCUS = AnnotationTypeDef("focus", "focus", Color.green, 15, 15)
    MAGNET = AnnotationTypeDef("magnets", "magnet", Color.green, 15, 15)
    LANDMARK = AnnotationTypeDef("landmarks", "landmark", Color.yellow, 15, 15)

    @classmethod
    def all(cls):
        """Returns all annotation types"""
        return [
            cls.SECTION,
            cls.ROI,
            cls.FOCUS,
            cls.MAGNET,
            cls.LANDMARK,
        ]

    @classmethod
    def all_but_landmark(cls):
        """Returns all annotation types except landmarks"""
        return [
            cls.SECTION,
            cls.ROI,
            cls.FOCUS,
            cls.MAGNET,
        ]

    @classmethod
    def section_annotations(cls):
        """Returns all annotation types except landmarks"""
        return [
            cls.ROI,
            cls.FOCUS,
            cls.MAGNET,
        ]


def transfer_wafer(wafer_1, wafer_2):
    """
    Transforms the annotation from one wafer instance to another one.
    The same ficucials must be defined in the two wafer instances.
    You should write some code to get it to work:
    wafer_1 = Wafer(path_1)
    wafer_2 = Wafer(path_2) ...
    """
    IJ.log(
        ("Transferring annotations from {} to {} ...".format(wafer_1, wafer_2)).center(
            100, "-"
        )
    )
    landmarks_1 = [wafer_1.landmarks[key].centroid for key in sorted(wafer_1.landmarks)]
    landmarks_2 = [wafer_2.landmarks[key].centroid for key in sorted(wafer_2.landmarks)]
    aff = GeometryCalculator.affine_t(
        [l[0] for l in landmarks_1],
        [l[1] for l in landmarks_1],
        [l[0] for l in landmarks_2],
        [l[1] for l in landmarks_2],
    )
    wafer_2.clear_annotations()
    for key in sorted(wafer_1.sections):
        for annotation_type in AnnotationType.all_but_landmark():
            if hasattr(wafer_1, annotation_type.name) and key in getattr(
                wafer_1, annotation_type.name
            ):
                wafer_2.add(
                    annotation_type,
                    GeometryCalculator.points_to_poly(
                        GeometryCalculator.xy_to_points(
                            *GeometryCalculator.apply_affine_t(
                                [
                                    p[0]
                                    for p in getattr(wafer_1, annotation_type.name)[
                                        key
                                    ].points
                                ],
                                [
                                    p[1]
                                    for p in getattr(wafer_1, annotation_type.name)[
                                        key
                                    ].points
                                ],
                                aff,
                            )
                        )
                    ),
                    key,
                )
    wafer_2.wafer_to_manager()
    wafer_2.save()
    IJ.log(
        (
            "Completed transfer of annotations from {} to {} ...".format(
                wafer_1, wafer_2
            )
        ).center(100, "-")
    )


class Wafer(object):
    def __init__(self, magc_path=None):
        self.mode = Mode.GLOBAL
        self.root, self.image_path = self.init_image_path()
        self.tsp_solver = TSPSolver(self.root)
        self.manager = init_manager()
        if magc_path is None:
            self.magc_path = self.init_magc_path()
        else:
            self.magc_path = magc_path
        self.init_images_global()
        self.sections = {}
        self.rois = {}
        self.focus = {}
        self.magnets = {}
        self.landmarks = {}
        self.transforms = {}
        self.poly_transforms = {}
        """poly_tranforms: global to local"""
        self.poly_transforms_inverse = {}
        """poly_tranforms_inverse: local to global"""
        # the serial order is the order in which the sections have been cut
        self.serialorder = []
        # the stageorder is the order that minimizes microscope stage travel
        # to image one section after the other
        self.stageorder = []
        self.GC = GeometryCalculator
        self.file_to_wafer()
        IJ.setTool("polygon")

    def __len__(self):
        """Returns the number of sections"""
        if not hasattr(self, "sections"):
            return 0
        return len(self.sections)

    @property
    def image(self):
        if self.mode is Mode.GLOBAL:
            return self.image_global
        return self.image_local

    @property
    def img(self):
        if self.mode is Mode.GLOBAL:
            return self.image_global
        return self.img_local

    @contextmanager
    def set_mode(self, mode):
        old_mode = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = old_mode

    @staticmethod
    def set_listeners():
        """Sets key and mouse wheel listeners"""
        add_key_listener_everywhere(KeyListener())
        add_mouse_wheel_listener_everywhere(MouseWheelListener())

    def set_global_mode(self):
        """useful when wafer accessed from another module"""
        self.mode = Mode.GLOBAL

    def set_local_mode(self):
        self.mode = Mode.LOCAL

    def init_image_path(self):
        """
        Finds the image used for navigation in magfinder.
        It is the image with the smallest size in the directory
        that does not contain "overview" in its name.
        """
        try:
            root = os.path.normpath(
                DirectoryChooser("Select the experiment folder.").getDirectory()
            )
        except Exception:
            IJ.showMessage("Exit", "There was a problem accessing the folder")
            sys.exit("No directory was selected. Exiting.")
        if not os.path.isdir(root):
            IJ.showMessage("Exit", "No directory was selected. Exiting.")
            sys.exit("No directory was selected. Exiting.")
        try:
            wafer_im_path = sorted(
                [
                    os.path.join(root, name)
                    for name in os.listdir(root)
                    if any(name.endswith(x) for x in ACCEPTED_IMAGE_FORMATS)
                    and not "verview" in name
                ],
                key=os.path.getsize,  # the smallest image is the one used for magfinder navigation
            )[0]
        except IndexError:
            IJ.showMessage(
                "Message",
                (
                    "There is no image (.tif, .png, .jpg, .jpeg, .tiff) in the experiment folder you selected."
                    + "\nAdd an image and start again the plugin."
                ),
            )
            sys.exit()
        return root, wafer_im_path

    def init_magc_path(self):
        """Loads existing .magc file or creates a new one if does not exist"""
        magc_paths = [
            os.path.join(self.root, filename)
            for filename in os.listdir(self.root)
            if filename.endswith(".magc")
        ]
        if not magc_paths:
            IJ.log("No .magc file found. Creating an empty one")
            wafer_name_from_user = get_name(
                "Please name this substrate",
                default_name="default_substrate",
                cancel_msg="The substrate needs a name. Using 'default_substrate'.",
            )
            if not wafer_name_from_user:
                wafer_name_from_user = "default_substrate"
            magc_path = os.path.join(self.root, "{}.magc".format(wafer_name_from_user))
            with open(magc_path, "w"):
                pass
        else:
            magc_path = magc_paths[0]
        return magc_path

    def file_to_wafer(self):
        """Populates the wafer instance from the .magc file"""
        start = System.nanoTime()
        config = ConfigParser.ConfigParser()
        with open(self.magc_path, "rb") as configfile:
            config.readfp(configfile)
        IJ.log(
            "Duration file_to_wafer 1: {}".format((System.nanoTime() - start) * 1e-9)
        )
        for header in config.sections():
            if "." in header:
                annotation_type, section_id, annotation_id = type_id(
                    header, delimiter_0="."
                )
                for key, val in config.items(header):
                    if key in ["polygon", "location"]:
                        vals = [float(x) for x in val.split(",")]
                        points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                        self.add(
                            annotation_type,
                            self.GC.points_to_poly(points),
                            (section_id, annotation_id)
                            if annotation_type is AnnotationType.ROI
                            else section_id,
                        )
            elif header in ["serialorder", "stageorder"]:
                if config.get(header, header) != "[]":
                    setattr(
                        self,
                        header,
                        [int(x) for x in config.get(header, header).split(",")],
                    )
        if not self.serialorder:
            self.serialorder = range(len(self.sections))
        IJ.log(
            (
                "File successfully read with \n{} sections \n{} rois \n{} focus"
                + "\n{} magnets \n{} landmarks"
            ).format(
                len(self.sections),
                len(self.rois),
                len(self.focus),
                len(self.magnets),
                len(self.landmarks),
            )
        )
        IJ.log("Duration file_to_wafer: {}".format((System.nanoTime() - start) * 1e-9))

    def wafer_to_manager(self):
        """Draws all rois from the wafer instance into the manager"""
        start = System.nanoTime()
        self.manager.reset()
        if self.mode is Mode.GLOBAL:
            self.wafer_to_manager_global()
        else:
            self.wafer_to_manager_local()
        IJ.log(
            "Duration wafer_to_manager: {:.2f} in mode {}".format(
                (System.nanoTime() - start) * 1e-9, self.mode
            )
        )

    def wafer_to_manager_global(self):
        for landmark in self.landmarks.values():
            self.manager.addRoi(landmark.poly)
            landmark.poly.setHandleSize(landmark.type_.handle_size_global)
        for section_id, section in sorted(self.sections.iteritems()):
            self.manager.addRoi(section.poly)
            section.poly.setHandleSize(section.type_.handle_size_global)
            for annotation_type in [
                AnnotationType.FOCUS,
                AnnotationType.MAGNET,
            ]:
                annotation = getattr(self, annotation_type.name).get(section_id)
                if annotation is not None:
                    self.manager.addRoi(annotation.poly)
                    annotation.poly.setHandleSize(annotation_type.handle_size_global)
            if section_id in self.rois:
                for _, subroi in sorted(self.rois[section_id].iteritems()):
                    self.manager.addRoi(subroi.poly)
                    subroi.poly.setHandleSize(annotation_type.handle_size_global)

    def wafer_to_manager_local(self):
        sorted_keys = sorted(self.sections)
        # sections are ordered in the local stack
        for id_o, o in enumerate(self.serialorder):
            section_id = sorted_keys[o]
            for annotation_type in [
                AnnotationType.SECTION,
                AnnotationType.FOCUS,
                AnnotationType.MAGNET,
            ]:
                annotation = getattr(self, annotation_type.name).get(section_id)
                if annotation is not None:
                    local_poly = self.GC.transform_points_to_poly(
                        annotation.points, self.poly_transforms[section_id]
                    )
                    local_poly.setName(str(annotation))
                    local_poly.setStrokeColor(annotation_type.color)
                    local_poly.setImage(self.image)
                    local_poly.setPosition(0, id_o + 1, 0)
                    local_poly.setHandleSize(annotation_type.handle_size_local)
                    self.manager.addRoi(local_poly)
            if section_id not in self.rois:
                continue
            for _, subroi in sorted(self.rois[section_id].iteritems()):
                local_poly = self.GC.transform_points_to_poly(
                    subroi.points, self.poly_transforms[section_id]
                )
                local_poly.setName(str(subroi))
                local_poly.setStrokeColor(subroi.type_.color)
                local_poly.setImage(self.image)
                local_poly.setPosition(0, id_o + 1, 0)
                local_poly.setHandleSize(subroi.type_.handle_size_local)
                self.manager.addRoi(local_poly)

    def clear_annotations(self):
        """Clears all annotations except the landmarks"""
        self.sections = {}
        self.rois = {}
        self.focus = {}
        self.magnets = {}

    def clear_transforms(self):
        """Clears all transforms"""
        self.transforms = {}
        self.poly_transforms = {}
        self.poly_transforms_inverse = {}

    def manager_to_wafer(self):
        """
        Populates the wafer from the roi manager
        Typically called after the user interacted with the UI
        """
        start = System.nanoTime()
        serial_order = copy.deepcopy(self.serialorder)
        self.clear_annotations()
        for roi in self.manager.iterator():
            annotation_type, section_id, annotation_id = type_id(roi.getName())
            self.add(
                annotation_type,
                roi,
                (section_id, annotation_id)
                if annotation_type is AnnotationType.ROI
                else section_id,
            )
        self.clear_transforms()
        self.compute_transforms()
        self.serialorder = serial_order  # needed here because serial order is changed when adding a section
        IJ.log(
            "Duration manager_to_wafer: {:.2f}".format(
                (System.nanoTime() - start) * 1e-9
            )
        )

    def save(self):
        """Saves the wafer annotations to the .magc file"""
        start = System.nanoTime()
        IJ.log("Saving ...")
        self.manager_to_wafer()
        config = ConfigParser.ConfigParser()
        for annotation_type in AnnotationType.all():
            if annotation_type is AnnotationType.ROI:
                continue
            annotations = getattr(self, annotation_type.name)
            if not annotations:
                continue
            config.add_section(annotation_type.name)
            config.set(
                annotation_type.name,
                "number",
                str(len(annotations)),
            )
            for _, annotation in sorted(annotations.iteritems()):
                header = annotation.header
                config.add_section(header)
                if annotation_type in [
                    AnnotationType.SECTION,
                    AnnotationType.ROI,
                    AnnotationType.FOCUS,
                ]:
                    config.set(
                        header,
                        "polygon",
                        self.GC.points_to_flat_string(annotation.points),
                    )
                    if annotation_type in [
                        AnnotationType.SECTION,
                        AnnotationType.ROI,
                    ]:
                        config.set(
                            header,
                            "center",
                            self.GC.point_to_flat_string(annotation.centroid),
                        )
                        config.set(header, "area", str(annotation.area))
                        config.set(
                            header,
                            "angle",
                            str(((annotation.angle - 90) % 360) - 180),
                        )
                elif annotation_type in [
                    AnnotationType.MAGNET,
                    AnnotationType.LANDMARK,
                ]:
                    config.set(
                        header,
                        "location",
                        self.GC.point_to_flat_string(annotation.centroid),
                    )
            config.add_section("end_{}".format(annotation_type.name))
        if self.rois:
            config.add_section(AnnotationType.ROI.name)
            config.set(
                AnnotationType.ROI.name,
                "number",
                sum([len(section_rois) for section_rois in self.rois.values()]),
            )
            for _, rois in sorted(self.rois.iteritems()):
                for _, roi in sorted(rois.iteritems()):
                    header = roi.header
                    config.add_section(header)
                    config.set(
                        header,
                        "polygon",
                        self.GC.points_to_flat_string(roi.points),
                    )
                    config.set(
                        header,
                        "center",
                        self.GC.point_to_flat_string(roi.centroid),
                    )
                    config.set(header, "area", str(roi.area))
                    config.set(
                        header,
                        "angle",
                        str(((roi.angle - 90) % 360) - 180),
                    )
        for order_name in ["serialorder", "stageorder"]:
            config.add_section(order_name)
            order = getattr(self, order_name)
            if not order:
                config.set(order_name, order_name, "[]")
            else:
                config.set(order_name, order_name, ",".join([str(x) for x in order]))

        with open(self.magc_path, "w") as configfile:
            config.write(configfile)
        IJ.log("Saved to {}".format(self.magc_path))
        self.save_csv()
        IJ.log("Duration save: {:.2f}".format((System.nanoTime() - start) * 1e-9))

    def save_csv(self):
        # TODO currently broken with multirois
        start = System.nanoTime()
        csv_path = os.path.join(self.root, "annotations.csv")
        with open(csv_path, "w") as f:
            f.write(
                ",".join(
                    [
                        "section_id",
                        "section_center_x",
                        "section_center_y",
                        "section_angle",
                        "roi_center_x",
                        "roi_center_y",
                        "roi_angle",
                        "magnet_x",
                        "magnet_y",
                        "landmark_x",
                        "landmark_y",
                        "stage_order",
                        "serial_order",
                    ]
                )
            )
            f.write("\n")
            for id, section_id in enumerate(sorted(self.sections.keys())):
                f.write(
                    ",".join(
                        [
                            str(x)
                            for x in [
                                section_id,
                                self.sections[section_id].centroid[0],
                                self.sections[section_id].centroid[1],
                                self.sections[section_id].angle,
                                self.rois[section_id][0].centroid[0]
                                if section_id in self.rois.keys()
                                else "",
                                self.rois[section_id][0].centroid[1]
                                if section_id in self.rois.keys()
                                else "",
                                self.rois[section_id][0].angle
                                if section_id in self.rois.keys()
                                else "",
                                self.magnets[section_id].centroid[0]
                                if section_id in self.magnets.keys()
                                else "",
                                self.magnets[section_id].centroid[1]
                                if section_id in self.magnets.keys()
                                else "",
                                self.landmarks[id].centroid[0]
                                if id in self.landmarks.keys()
                                else "",
                                self.landmarks[id].centroid[1]
                                if id in self.landmarks.keys()
                                else "",
                                self.stageorder[id]
                                if len(self.stageorder) > id
                                else "",
                                self.serialorder[id]
                                if len(self.serialorder) > id
                                else "",
                            ]
                        ],
                    )
                )
                # unusual case: if there are more landmarks than sections
                for i in range(len(self.sections), len(self.landmarks)):
                    f.write(
                        ",,,,,,,,,{},{},,".format(
                            self.landmarks[i].centroid[0], self.landmarks[i].centroid[1]
                        )
                    )
                f.write("\n")
        IJ.log("Annotations saved to {}".format(csv_path))
        IJ.log("Duration save_csv: {:.2f}".format((System.nanoTime() - start) * 1e-9))

    def close_mode(self):
        """
        1.Saves the current wafer
        2.Closes the current display mode
        3.Restores the standard GUI key listeners
        """
        self.save()
        if self.mode is Mode.GLOBAL:
            self.image.hide()
        else:
            self.image.close()
        self.manager.reset()
        # restore the keylisteners of the IJ window
        map(IJ.getInstance().addKeyListener, initial_ij_key_listeners)

    def start_local_mode(self):
        """Starts local display mode"""
        start = System.nanoTime()
        IJ.log("Starting local mode ...")
        self.mode = Mode.LOCAL
        self.compute_transforms()
        self.create_local_stack()
        self.wafer_to_manager()
        self.set_listeners()
        self.manager.runCommand("UseNames", "false")
        self.manager.runCommand("Show None")
        set_roi_and_update_roi_manager(0)  # select first ROI
        self.arrange_windows()
        IJ.log(
            "Duration start_local_mode: {:.2f}".format(
                (System.nanoTime() - start) * 1e-9
            )
        )

    def compute_transforms(self):
        """
        1.self.transforms[section_key] transforms the global wafer image
        to an image in which the section section_key is centered at 0
        and has an angle of 0 degrees
        2.the poly_transforms are almost like the self.transforms except that
        they contain an offset due to the fact that an ImagePlus is displayed
        with their top-left corner at 0,0 and not at -w/2,-h/2
        """
        start = System.nanoTime()
        _, _, display_size, _ = self.get_display_parameters()
        self.local_display_size = display_size
        for section_id, section in self.sections.iteritems():
            # image transform
            aff = AffineTransform2D()
            aff.translate([-v for v in section.centroid])
            aff.rotate(section.angle * Math.PI / 180.0)
            self.transforms[section_id] = aff
            # poly transform (there is an offset)
            aff_copy = aff.copy()
            poly_translation = AffineTransform2D()
            poly_translation.translate([0.5 * v for v in self.local_display_size])
            self.poly_transforms[section_id] = aff_copy.preConcatenate(poly_translation)
            self.poly_transforms_inverse[section_id] = self.poly_transforms[
                section_id
            ].inverse()
        IJ.log(
            "Duration compute_transforms: {:.2f}".format(
                (System.nanoTime() - start) * 1e-9
            )
        )

    def create_local_stack(self):
        """Creates the local stack with imglib2 framework"""
        start = System.nanoTime()
        display_params = (
            [-intr(0.5 * v) for v in self.local_display_size],
            [intr(0.5 * v) for v in self.local_display_size],
        )
        sorted_section_keys = sorted(self.sections)
        imgs = [
            Views.interval(
                RV.transform(
                    Views.interpolate(
                        Views.extendZero(self.img_global),
                        # NLinearInterpolatorFactory()
                        NearestNeighborInterpolatorFactory(),
                    ),
                    self.transforms[sorted_section_keys[o]],
                ),
                display_params[0],
                display_params[1],
            )
            for o in self.serialorder
        ]
        self.img_local = Views.permute(
            Views.addDimension(Views.stack(imgs), 0, 0), 3, 2
        )
        IL.show(self.img)
        self.image_local = IJ.getImage()

        IJ.log(
            "Duration create_local_stack: {:.2f}".format(
                (System.nanoTime() - start) * 1e-9
            )
        )

    def add(self, annotation_type, poly, annotation_id):
        """Adds an annotation to the wafer and returns it"""
        if annotation_type is AnnotationType.ROI:
            section_id = annotation_id[0]
        else:
            section_id = annotation_id
        if (
            annotation_type is AnnotationType.SECTION
            and annotation_id not in self.sections
        ):
            # appends to serial order only if new section
            self.serialorder.append(len(self))
        if self.mode is Mode.GLOBAL:
            annotation = Annotation(
                annotation_type,
                poly,
                annotation_id,
            )
        else:
            # transform to global coordinates when adding from local mode
            annotation = Annotation(
                annotation_type,
                self.GC.transform_points_to_poly(
                    self.GC.poly_to_points(poly),
                    self.poly_transforms_inverse[section_id],
                ),
                annotation_id,
            )
        if annotation_type is AnnotationType.ROI:
            if self.rois.get(section_id) is None:
                self.rois[section_id] = {}
            self.rois[section_id][annotation_id[1]] = annotation
        else:
            getattr(self, annotation_type.name)[section_id] = annotation
        return annotation

    def add_section(self, poly, annotation_id):
        return self.add(AnnotationType.SECTION, poly, annotation_id)

    def add_roi(self, poly, annotation_id):
        return self.add(AnnotationType.ROI, poly, annotation_id)

    def add_magnet(self, poly, annotation_id):
        return self.add(AnnotationType.MAGNET, poly, annotation_id)

    def add_focus(self, poly, annotation_id):
        return self.add(AnnotationType.FOCUS, poly, annotation_id)

    def remove_current(self):
        if self.mode is Mode.GLOBAL:
            return
        selected_indexes = self.manager.getSelectedIndexes()
        if len(selected_indexes) != 1:
            IJ.showMessage(
                "Warning",
                "To delete an annotation with [x], one and only one annotation"
                " must be selected in blue in the annotation manager",
            )
            return
        selected_poly = self.manager.getRoi(selected_indexes[0])
        poly_name = selected_poly.getName()
        annotation_type, section_id, annotation_id = type_id(poly_name)
        self.save()
        if annotation_type in {
            AnnotationType.MAGNET,
            AnnotationType.ROI,
            AnnotationType.FOCUS,
        }:
            if get_OK("Delete {}?".format(poly_name)):
                delete_selected_roi()
                self.image.killRoi()
                if annotation_type is AnnotationType.ROI:
                    del self.rois[section_id][annotation_id]
                del getattr(self, annotation_type.name)[annotation_id]
                # select the section
                self.manager.select(
                    get_roi_index_by_name(str(self.sections[section_id]))
                )
        elif annotation_type is AnnotationType.SECTION:
            # deleting a sections also deletes the linked annotations (roi(s),focus,magnet)
            section_id_manager = get_roi_index_by_name(str(self.sections[section_id]))
            linked_annotations = []
            message = ""
            # build the message by screening all existing linked annotations
            for type_ in [
                AnnotationType.FOCUS,
                AnnotationType.MAGNET,
            ]:
                linked_annotation = getattr(self, type_.name).get(annotation_id)
                if linked_annotation is not None:
                    message += "{}\n \n".format(str(linked_annotation))
                    linked_annotations.append(linked_annotation)
            if section_id in self.rois:
                for _, subroi in sorted(self.rois[section_id].iteritems()):
                    message += "{}\n \n".format(str(subroi))
                    linked_annotations.append(subroi)
            message = "".join(
                [
                    "Delete {}?".format(poly_name),
                    "\n \nIt will also delete\n \n" if message else "",
                    message,
                ]
            )
            if not get_OK(message):
                return
            if self.image.getNSlices() == 1:
                if get_OK(
                    "Case not yet handled: you are trying to delete the only existing section."
                    "\n\nThe plugin will close. Please delete the .magc file"
                    " and start over from scratch instead.\n\nContinue?"
                ):
                    self.image.close()
                    self.manager.close()
                    sys.exit()
                else:
                    return
            self.image.killRoi()
            # delete linked annotations in manager and in wafer
            for linked_annotation in linked_annotations:
                index = get_roi_index_by_name(str(linked_annotation))
                delete_roi_by_index(index)
                del getattr(self, linked_annotation.type_.name)[section_id]
            # delete section in manager
            section_id_manager = get_roi_index_by_name(str(self.sections[section_id]))
            delete_roi_by_index(section_id_manager)
            # rearrange serial order
            # 1. delete the serialorder entry of that section
            del self.serialorder[sorted(self.sections.keys()).index(section_id)]
            # 2. decrements the serialorder id of the sections with an id greater than
            # the one that was deleted
            self.serialorder = [
                o - 1 if (o > sorted(self.sections.keys()).index(section_id)) else o
                for o in self.serialorder
            ]

            del self.sections[section_id]
            del self.transforms[section_id]
            del self.poly_transforms[section_id]
            del self.poly_transforms_inverse[section_id]

            self.image.close()
            self.manager.reset()

            self.start_local_mode()

            # select the next section
            if section_id_manager < self.manager.getCount():
                set_roi_and_update_roi_manager(section_id_manager)
            else:
                set_roi_and_update_roi_manager(self.manager.getCount() - 1)

    def init_images_global(self):
        self.image_global = IJ.openImage(self.image_path)
        self.img_global = IL.wrap(self.image)

    def arrange_windows(self):
        IJ.getInstance().setLocation(
            0, IJ.getScreenSize().height - IJ.getInstance().getSize().height
        )
        # enlarge window, select first ROI, get polygon tool
        win = self.image.getWindow()
        if self.image.getNSlices() > 1:
            win.remove(win.getComponents()[1])  # remove slider bar
        win.maximize()
        win.setLocation(
            IJ.getScreenSize().width
            - self.manager.getSize().width
            - win.getSize().width,
            0,
        )
        # adjust focus
        self.manager.toFront()
        time.sleep(0.05)
        IJ.selectWindow(self.image.getTitle())

    def start_global_mode(self):
        start = System.nanoTime()
        self.mode = Mode.GLOBAL
        IJ.run("Labels...", "color=white font=10 use draw")
        self.image.show()
        self.arrange_windows()
        self.wafer_to_manager()
        self.set_listeners()
        time.sleep(0.1)
        self.manager.runCommand("Show All")
        # self.manager.runCommand("Show All with labels")
        IJ.run("Labels...", "color=white font=10 use draw")
        self.manager.runCommand("Show All without labels")
        IJ.log(
            "Duration start_global_mode: {:.2f}".format(
                (System.nanoTime() - start) * 1e-9
            )
        )

    def suggest_annotation_ids(self, annotation_type):
        item_ids = [item.id_ for item in getattr(self, annotation_type.name).values()]
        return suggest_ids(item_ids)

    def get_closest(self, annotation_type, point):
        """Get the closest annotations of a certain type closest to a given point"""
        distances = sorted(
            [
                [self.GC.get_distance(point, item.centroid), item]
                for item in getattr(self, annotation_type.name).values()
            ]
        )
        return [d[1] for d in distances]

    def get_display_parameters(self):
        """calculate [display_size, crop_size, tissue_magnet_distance] based on sectionSize"""
        tissue_magnet_distance = 0

        section_extent = 0
        for section in self.sections.values()[: min(5, len(self.sections))]:
            section_extent = max(
                section_extent, self.GC.longest_diagonal(section.points)
            )

        display_size = (
            [intr(DISPLAY_FACTOR * section_extent)] * 2
            if not self.magnets
            else [intr(DISPLAY_FACTOR * 1.2 * section_extent)] * 2
        )

        display_center = [
            intr(0.5 * display_size[0]),
            intr(0.5 * display_size[1]),
        ]
        crop_size = [2 * display_size[0], 2 * display_size[1]]

        if self.magnets:
            tissue_magnet_distances = []
            for magnet_id in sorted(self.magnets.keys()[: min(5, len(self.magnets))]):
                tissue_magnet_distances.append(
                    self.GC.get_distance(
                        self.sections[magnet_id].centroid,
                        self.magnets[magnet_id].centroid,
                    )
                )
            tissue_magnet_distance = sum(tissue_magnet_distances) / len(
                tissue_magnet_distances
            )
        return display_center, tissue_magnet_distance, display_size, crop_size

    def compute_stage_order(self):
        """Computes the stage order by solving a TSP on the section centroids"""
        center_points = [
            pFloat(*wafer.sections[section_key].centroid)
            for section_key in sorted(self.sections)
        ]
        # fill distance matrix
        distances = TSPSolver.init_mat(len(self.sections), initValue=999999)
        for a, b in itertools.combinations_with_replacement(
            range(len(self.sections)), 2
        ):
            distances[a][b] = distances[b][a] = center_points[a].distance(
                center_points[b]
            )
        self.stageorder = self.tsp_solver.compute_tsp_order(distances)

    def update_section(self, section_id, transform, ref_section_points, ref_roi_points):
        """
        Updates the location of a section by applying a transform to a reference section/roi.
        Typically used by the MagReorderer after transforms have been found
        in order to stack the sections in serial order
        """
        with self.set_mode(Mode.GLOBAL):
            self.add_section(
                self.GC.transform_points_to_poly(
                    ref_section_points,
                    transform,
                ),
                section_id,
            )
            self.add_roi(
                self.GC.transform_points_to_poly(ref_roi_points, transform),
                (section_id, 0),
            )

    def renumber_sections(self):
        """Renumbering the sections to have consecutive numbers without gaps:
        "0,1,4,5,7 -> 0,1,2,3,4"
        """
        current_serial_order = copy.deepcopy(self.serialorder)
        if self.mode is Mode.LOCAL:
            IJ.log(
                "Closing local mode. Section renumbering will be done in global mode"
            )
            self.close_mode()
            self.start_global_mode()
        for new_key, key in enumerate(sorted(self.sections)):
            if new_key == key:
                continue
            for annotation_type in [
                AnnotationType.SECTION,
                AnnotationType.FOCUS,
                AnnotationType.MAGNET,
            ]:
                annotation = getattr(self, annotation_type.name).get(key)
                if annotation is not None:
                    self.add(annotation_type, annotation.poly, new_key)
                    del getattr(self, annotation_type.name)[key]
            if not self.rois:  # TODO needed?
                continue
            for subroi_id, subroi in self.rois[key]:
                self.add_roi(subroi.poly, (new_key, subroi_id))
                del self.rois[key][subroi_id]

        self.clear_transforms()
        self.compute_transforms()
        self.wafer_to_manager()
        self.serialorder = current_serial_order
        IJ.log("{} sections have been renumbered".format(len(self)))

    def suggest_roi_ids(self, section_id):
        if not self.rois:
            return ["roi-{:04}.{:02}".format(section_id, 0)]
        section_rois = self.rois.get(section_id)
        if section_rois is None:
            return ["roi-{:04}.{:02}".format(section_id, 0)]
        return [
            "roi-{:04}.{:02}".format(section_id, roi_id_suggestion)
            for roi_id_suggestion in sorted(
                set(suggest_ids(section_rois.keys())).union(set(section_rois))
            )
        ]

    def get_section_id_from_local_position(self):
        """
        While the local stack is open, returns the section_id corresponding
        to the current annotation being displayed
        """
        slice_id = wafer.image.getSlice()
        return sorted(wafer.sections.keys())[wafer.serialorder[slice_id - 1]]


class Annotation(object):
    def __init__(self, annotation_type, poly, id_):
        self.type_ = annotation_type
        self.poly = poly
        self.id_ = id_
        self.points = GeometryCalculator.poly_to_points(poly)
        self.centroid = self.compute_centroid()
        self.area = self.compute_area()
        self.angle = self.compute_angle()
        self.set_poly_properties()

    def __str__(self):
        if self.type_ is AnnotationType.ROI:
            return "{}-{:04}.{:02}".format(self.type_.string, *self.id_)
        return "{}-{:04}".format(self.type_.string, self.id_)

    def __len__(self):
        return self.poly.size()

    @property
    def header(self):
        if self.type_ is AnnotationType.ROI:
            return "{}.{:04}.{:02}".format(self.type_.string, *self.id_)
        return "{}.{:04}".format(self.type_.string, self.id_)

    def compute_area(self):
        """Simply take bounding box area. getStatistics.pixelcount was too slow"""
        bounds = self.poly.getBounds()
        return bounds.getHeight() * bounds.getWidth()

    def set_poly_properties(self):
        self.poly.setName(str(self))
        self.poly.setStrokeColor(self.type_.color)

    def contains(self, point):
        return self.poly.containsPoint(*point)

    def compute_centroid(self):
        if len(self) == 1:
            return self.points[0]
        return list(self.poly.getContourCentroid())

    def compute_angle(self):
        if len(self) < 2:
            return None
        return self.poly.getFloatAngle(
            self.points[0][0],
            self.points[0][1],
            self.points[1][0],
            self.points[1][1],
        )


class GeometryCalculator(object):
    @staticmethod
    def get_distance(p_1, p_2):
        """Distance between two points"""
        return Math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)

    @classmethod
    def longest_diagonal(cls, points):
        """Longest pairwise distance among all points"""
        max_diag = 0
        for p1, p2 in itertools.combinations(points, 2):
            max_diag = max(cls.get_distance(p1, p2), max_diag)
        return max_diag

    @staticmethod
    def points_to_poly(points):
        """From list of points to Fiji pointroi or polygonroi"""
        if len(points) == 1:
            return PointRoi(*[float(v) for v in points[0]])
        if len(points) == 2:
            polygon_type = PolygonRoi.POLYLINE
        elif len(points) > 2:
            polygon_type = PolygonRoi.POLYGON
        return PolygonRoi(
            [float(point[0]) for point in points],
            [float(point[1]) for point in points],
            polygon_type,
        )

    @staticmethod
    def poly_to_points(poly):
        float_polygon = poly.getFloatPolygon()
        return [[x, y] for x, y in zip(float_polygon.xpoints, float_polygon.ypoints)]

    @staticmethod
    def points_to_flat_string(points):
        """[[1,2],[3,4]] -> 1,2,3,4"""
        points_flat = []
        for point in points:
            points_flat.append(point[0])
            points_flat.append(point[1])
        points_string = ",".join([str(round(x, 3)) for x in points_flat])
        return points_string

    @staticmethod
    def point_to_flat_string(point):
        """[1,2] -> 1,2"""
        flat_string = ",".join([str(round(x, 3)) for x in point])
        return flat_string

    @staticmethod
    def points_to_xy(points):
        """[[1,2],[3,4]] -> [(1,3),(2,4)]"""
        return zip(*points)

    @staticmethod
    def xy_to_points(x, y):
        return zip(x, y)

    @staticmethod
    def transform_points(source_points, aff):
        target_points = []
        for source_point in source_points:
            target_point = jarray.array([0, 0], "d")
            aff.apply(source_point, target_point)
            target_points.append(target_point)
        return target_points

    @staticmethod
    def to_imglib2_aff(trans):
        mat_data = jarray.array([[0, 0, 0], [0, 0, 0]], java.lang.Class.forName("[D"))
        trans.toMatrix(mat_data)
        imglib2_transform = AffineTransform2D()
        imglib2_transform.set(mat_data)
        return imglib2_transform

    @classmethod
    def transform_points_to_poly(cls, source_points, aff):
        target_points = cls.transform_points(source_points, aff)
        return cls.points_to_poly(target_points)

    @staticmethod
    def affine_t(x_in, y_in, x_out, y_out):
        """Fits an affine transform to given points"""
        X = Matrix(
            jarray.array(
                [[x, y, 1] for (x, y) in zip(x_in, y_in)], java.lang.Class.forName("[D")
            )
        )
        Y = Matrix(
            jarray.array(
                [[x, y, 1] for (x, y) in zip(x_out, y_out)],
                java.lang.Class.forName("[D"),
            )
        )
        aff = X.solve(Y)
        return aff

    @staticmethod
    def apply_affine_t(x_in, y_in, aff):
        X = Matrix(
            jarray.array(
                [[x, y, 1] for (x, y) in zip(x_in, y_in)], java.lang.Class.forName("[D")
            )
        )
        Y = X.times(aff)
        x_out = [float(y[0]) for y in Y.getArrayCopy()]
        y_out = [float(y[1]) for y in Y.getArrayCopy()]
        return x_out, y_out

    @staticmethod
    def invert_affine_t(aff):
        return aff.inverse()

    @staticmethod
    def rigid_t(x_in, y_in, x_out, y_out):
        rigidModel = RigidModel2D()
        pointMatches = HashSet()
        for x_i, y_i, x_o, y_o in zip(x_in, y_in, x_out, y_out):
            pointMatches.add(
                PointMatch(
                    Point([x_i, y_i]),
                    Point([x_o, y_o]),
                )
            )
        rigidModel.fit(pointMatches)
        return rigidModel

    @staticmethod
    def apply_rigid_t(x_in, y_in, rigid_model):
        x_out = []
        y_out = []
        for x_i, y_i in zip(x_in, y_in):
            x_o, y_o = rigid_model.apply([x_i, y_i])
            x_out.append(x_o)
            y_out.append(y_o)
        return x_out, y_out

    @classmethod
    def get_imglib2_transform_scaling(cls, transform):
        s1 = [0, 0]
        s2 = [1000, 1000]
        t1, t2 = cls.transform_points([s1, s2], transform)
        return cls.get_distance(t1, t2) / float(cls.get_distance(s1, s2))

    # @classmethod
    # def apply_t(x_in, y_in, transform):
    #     if len(source_section) == 2:
    #         compute_t = cls.rigid_t
    #         apply_t = cls.apply_rigid_t
    #     else:
    #         compute_t = cls.affine_t
    #         apply_t = cls.apply_affine_t

    @classmethod
    def propagate_points(cls, source_section, source_points, target_section):
        """Transforms points linked to a source section to a target section"""
        source_section_x, source_section_y = cls.points_to_xy(source_section)
        source_points_x, source_points_y = cls.points_to_xy(source_points)
        target_section_x, target_section_y = cls.points_to_xy(target_section)
        if len(source_section) == 2:
            compute_t = cls.rigid_t
            apply_t = cls.apply_rigid_t
        else:
            compute_t = cls.affine_t
            apply_t = cls.apply_affine_t
        trans = compute_t(
            source_section_x, source_section_y, target_section_x, target_section_y
        )
        target_points_x, target_points_y = apply_t(
            source_points_x, source_points_y, trans
        )
        target_points = [[x, y] for x, y in zip(target_points_x, target_points_y)]
        return target_points


class TSPSolver(object):
    def __init__(self, root):
        self.concorde_path, self.linkern_path = self.get_tsp_solver_paths()
        self.root = root

    def get_tsp_solver_paths(self):
        """
        get concorde solver, linkern solver, cygwin1.dll
        try to download if not present
        """
        plugin_folder = IJ.getDirectory("plugins")
        cygwindll_path = os.path.join(plugin_folder, "cygwin1.dll")
        linkern_path = os.path.join(plugin_folder, "linkern.exe")
        concorde_path = os.path.join(plugin_folder, "concorde.exe")

        concorde_url = r"https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/cygwin/concorde.exe.gz"
        linkern_url = r"https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/cygwin/linkern.exe.gz"
        cygwindll_url = (
            r"https://raw.githubusercontent.com/templiert/MagFinder/master/cygwin1.dll"
        )
        download_msg = (
            "Downloading {} to the Fiji plugins directory from \n \n{}\n \n"
            "The file is needed to compute the 'stage order' that minimizes stage travel"
            "\nand the serial order of the sections."
            "\n \nDo you agree? "
        )
        if "windows" in System.getProperty("os.name").lower():
            # download cygwin1.dll
            if not os.path.isfile(cygwindll_path):
                # download cygwin1.dll
                if get_OK(
                    download_msg.format(
                        "Windows-10 precompiled cygwin1.dll", cygwindll_url
                    ),
                    window_name="Download?",
                ):
                    try:
                        FileUtils.copyURLToFile(
                            URL(cygwindll_url), File(cygwindll_path)
                        )
                    except (Exception, java_exception) as e:
                        IJ.log("Failed to download cygwin1.dll due to {}".format(e))
            # download concorde and linkern solvers
            for path, url in zip(
                [concorde_path, linkern_path], [concorde_url, linkern_url]
            ):
                if not os.path.isfile(path):
                    if get_OK(
                        download_msg.format("solver for traveling salesman", url),
                        window_name="Download?",
                    ):
                        self.download_unzip(url, path)
        if not all(
            (os.path.isfile(p) for p in (cygwindll_path, concorde_path, linkern_path))
        ):
            concorde_path = linkern_path = cygwindll_path = None
        return concorde_path, linkern_path

    @staticmethod
    def download_unzip(url, target_path):
        # download, unzip, clean
        IJ.log("Downloading TSP solver from " + str(url))
        gz_path = os.path.join(IJ.getDirectory("plugins"), "temp.gz")
        try:
            FileUtils.copyURLToFile(URL(url), File(gz_path))
            gis = GZIPInputStream(FileInputStream(gz_path))
            Files.copy(gis, Paths.get(target_path))
            gis.close()
            os.remove(gz_path)
        except (Exception, java_exception) as e:
            IJ.log("Failed to download from " + str(url) + " due to " + str(e))

    def order_from_mat(self, mat, root_folder, solver_path, solution_name=""):
        tsplib_path = os.path.join(root_folder, "TSPMat.tsp")
        self.save_mat_to_TSPLIB(mat, tsplib_path)
        solution_path = os.path.join(
            root_folder, "solution_{}.txt".format(solution_name)
        )
        if os.path.isfile(solution_path):
            os.remove(solution_path)
        command = '"{}" -o "{}" "{}"'.format(solver_path, solution_path, tsplib_path)
        # command = (
        #     '"' + solver_path + '" -o "' + solution_path + '" "' + tsplib_path + '"'
        # )
        # IJ.log('TSP solving command ' + str(command))
        Runtime.getRuntime().exec(command)

        while not os.path.isfile(solution_path):
            time.sleep(1)
            IJ.log(
                "Computing TSP solution with the {} solver...".format(
                    os.path.basename(solver_path).replace(".exe", "")
                )
            )
        time.sleep(1)

        if "linkern.exe" in solver_path:
            with open(solution_path, "r") as f:
                order = [int(line.split(" ")[0]) for line in f.readlines()[1:]]
        elif "concorde.exe" in solver_path:
            with open(solution_path, "r") as f:
                lines = f.readlines()
                order = [int(x) for x in lines[1].replace(" \n", "").split(" ")]
        # remove the dummy city 0 and apply a -1 offset
        order.remove(0)
        order = [o - 1 for o in order]

        for name, cost in zip(
            ("non-", ""),
            (
                sum([mat[t][t + 1] for t in range(len(order) - 1)]),
                sum([mat[o1][o2] for o1, o2 in pairwise(order)]),
            ),
        ):
            IJ.log(
                "The total cost of the {}optimized order is {} (a.u.)".format(
                    name, intr(cost)
                )
            )
        # delete temporary files
        for p in [solution_path, tsplib_path]:
            if os.path.isfile(p):
                os.remove(p)
        return order

    @staticmethod
    def save_mat_to_TSPLIB(mat, path):
        with open(path, "w") as f:
            f.write("NAME: Section_Similarity_Data\n")
            f.write("TYPE: TSP\n")
            f.write("DIMENSION: {}\n".format(len(mat) + 1))
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
            f.write("NODE_COORD_TYPE: NO_COORDS\n")
            f.write("DISPLAY_DATA_TYPE: NO_DISPLAY\n")
            f.write("EDGE_WEIGHT_SECTION\n")

            distances = [0] * len(mat)  # dummy city
            for i, j in itertools.combinations(range(len(mat)), 2):
                distance = mat[i][j]
                distances.append(intr(distance))
            for id, distance in enumerate(distances):
                f.write(str(distance))
                if (id + 1) % 10 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
            f.write("EOF" + "\n")

    @staticmethod
    def init_mat(n, initValue=0):
        a = Array.newInstance(java.lang.Float, [n, n])
        for i, j in itertools.product(range(n), repeat=2):
            a[i][j] = initValue
        return a

    def compute_tsp_order(self, pairwise_costs):
        if len(pairwise_costs) < 2:
            return
        if len(pairwise_costs) < 8:
            solver_path = self.concorde_path
        else:
            solver_path = self.linkern_path
        if not solver_path:
            IJ.log(
                "Could not compute the stage-movement-minimizing order"
                " because the solver or cygwin1.dll are missing"
            )
            return
        try:
            order = self.order_from_mat(pairwise_costs, self.root, solver_path)
            IJ.log("The optimal order is: {}".format(order))
        except (Exception, java_exception) as e:
            IJ.log("The order could not be computed: {}".format(e))
            return []
        return order


# ----- Listeners and handlers ----- #
def add_key_listener_everywhere(my_listener):
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
        elem.addKeyListener(my_listener)

    for id0, comp0 in enumerate(get_roi_manager().getComponents()):
        for kl in comp0.getKeyListeners():
            comp0.removeKeyListener(kl)
        comp0.addKeyListener(my_listener)

        for id1, comp1 in enumerate(comp0.getComponents()):
            # if (type(comp1) == Button) and (comp1.getLabel() != 'Delete'):
            if type(comp1) == Button:
                comp0.remove(comp1)
            elif type(comp1) == Checkbox and (wafer.mode is Mode.LOCAL):
                comp0.remove(comp1)
            else:
                for kl in comp1.getKeyListeners():
                    comp1.removeKeyListener(kl)
                comp1.addKeyListener(my_listener)
                try:
                    for id2, comp2 in enumerate(comp1.getComponents()):
                        for kl in comp2.getKeyListeners():
                            comp2.removeKeyListener(kl)
                        comp2.addKeyListener(my_listener)
                except:
                    pass


def add_mouse_wheel_listener_everywhere(my_listener):
    for elem in (
        [
            IJ.getImage().getWindow(),
            IJ.getImage().getWindow().getCanvas(),
        ]
        # ui.getDefaultUI().getConsolePane().getComponent(),]
        # IJ.getInstance]
        + list(WindowManager.getAllNonImageWindows())
    ):
        kls = elem.getMouseWheelListeners()
        map(elem.removeMouseWheelListener, kls)
        elem.addMouseWheelListener(my_listener)

    for id0, comp0 in enumerate(get_roi_manager().getComponents()):
        for kl in comp0.getMouseWheelListeners():
            comp0.removeMouseWheelListener(kl)
        comp0.addMouseWheelListener(my_listener)

        for id1, comp1 in enumerate(comp0.getComponents()):
            # if (type(comp1) == Button) and (comp1.getLabel() != 'Delete'):
            if type(comp1) == Button:
                comp0.remove(comp1)
            else:
                for kl in comp1.getMouseWheelListeners():
                    comp1.removeMouseWheelListener(kl)
                comp1.addMouseWheelListener(my_listener)
                try:
                    for id2, comp2 in enumerate(comp1.getComponents()):
                        for kl in comp2.getMouseWheelListeners():
                            comp2.removeMouseWheelListener(kl)
                        comp2.addMouseWheelListener(my_listener)
                except:
                    pass


class ButtonClick(ActionListener):
    def actionPerformed(self, event):
        source = event.getSource()
        string_field = source.getParent().getStringFields()[0]
        string_field.setText(source.label.split(" ")[-1])


# --- Mouse wheel listeners --- #
class MouseWheelListener(MouseAdapter):
    def mouseWheelMoved(self, mouseWheelEvent):
        if wafer.mode is Mode.GLOBAL:
            handle_mouse_wheel_global(mouseWheelEvent)
        elif wafer.mode is Mode.LOCAL:
            handle_mouse_wheel_local(mouseWheelEvent)


def handle_mouse_wheel_global(mouseWheelEvent):
    mouseWheelEvent.consume()
    if mouseWheelEvent.isShiftDown():
        if mouseWheelEvent.getWheelRotation() == 1:
            move_fov("right")
        else:
            move_fov("left")
    elif (not mouseWheelEvent.isShiftDown()) and (not mouseWheelEvent.isControlDown()):
        if mouseWheelEvent.getWheelRotation() == 1:
            move_fov("down")
        else:
            move_fov("up")
    elif mouseWheelEvent.isControlDown():
        if mouseWheelEvent.getWheelRotation() == 1:
            IJ.run("Out [-]")
        elif mouseWheelEvent.getWheelRotation() == -1:
            IJ.run("In [+]")


def handle_mouse_wheel_local(mouseWheelEvent):
    mouseWheelEvent.consume()
    if mouseWheelEvent.isControlDown():
        move_roi_manager_selection(10 * mouseWheelEvent.getWheelRotation())
    else:
        move_roi_manager_selection(mouseWheelEvent.getWheelRotation())


# --- End Mouse wheel listeners --- #

# --- Key listeners --- #
class KeyListener(KeyAdapter):
    def keyPressed(self, event):
        keycode = event.getKeyCode()
        event.consume()
        if event.getKeyCode() == KeyEvent.VK_J and wafer.mode is Mode.GLOBAL:
            reorderer = MagReorderer.MagReorderer(wafer)
            reorderer.reorder()
        elif keycode == KeyEvent.VK_S:
            wafer.save()
        elif keycode == KeyEvent.VK_Q:  # terminate and save
            wafer.manager_to_wafer()  # will be repeated in close_mode but it's OK
            wafer.compute_stage_order()
            wafer.close_mode()
            wafer.manager.close()
        elif keycode == KeyEvent.VK_O:
            wafer.manager_to_wafer()
            wafer.compute_stage_order()
            wafer.save()
        elif wafer.mode is Mode.GLOBAL:
            handle_key_global(event)
        elif wafer.mode is Mode.LOCAL:
            handle_key_local(event)


def handle_key_global(keyEvent):
    keycode = keyEvent.getKeyCode()
    if keycode == KeyEvent.VK_A:
        handle_key_a()
    if keycode == KeyEvent.VK_N:
        wafer.renumber_sections()
    if keycode == KeyEvent.VK_T:
        if wafer.sections:
            wafer.close_mode()
            wafer.start_local_mode()
        else:
            IJ.showMessage(
                "Cannot toggle to local mode because there are no sections defined."
            )
    if keycode == KeyEvent.VK_H:
        IJ.showMessage("Help for global mode", HELP_MSG_GLOBAL)
    if keycode == KeyEvent.VK_1:
        toggle_fill(AnnotationType.SECTION)
    if keycode == KeyEvent.VK_2:
        toggle_fill(AnnotationType.ROI)
    if keycode == KeyEvent.VK_3:
        toggle_fill(AnnotationType.FOCUS)
    if keycode == KeyEvent.VK_0:
        toggle_labels()
    if keycode == KeyEvent.VK_EQUALS:
        IJ.run("In [+]")
    if keycode == KeyEvent.VK_MINUS:
        IJ.run("Out [-]")
    if keycode == KeyEvent.VK_UP:
        move_fov("up")
    if keycode == KeyEvent.VK_DOWN:
        move_fov("down")
    if keycode == KeyEvent.VK_RIGHT:
        move_fov("right")
    if keycode == KeyEvent.VK_LEFT:
        move_fov("left")
    if keycode == KeyEvent.VK_M:
        handle_key_m_global()


def handle_key_local(keyEvent):
    keycode = keyEvent.getKeyCode()
    manager = get_roi_manager()
    if keycode == KeyEvent.VK_A:
        handle_key_a()
    if keycode == KeyEvent.VK_X:
        handle_key_x_local()
    if keycode == KeyEvent.VK_P:
        handle_key_p_local()
    if keycode == KeyEvent.VK_M:
        handle_key_m_local()
    if keycode == KeyEvent.VK_D:
        move_roi_manager_selection(-1)
    if keycode == KeyEvent.VK_F:
        move_roi_manager_selection(1)
    if keycode == KeyEvent.VK_C:
        move_roi_manager_selection(-10)
    if keycode == KeyEvent.VK_V:
        move_roi_manager_selection(10)
    if keycode == KeyEvent.VK_E:
        selectedIndex = manager.getSelectedIndex()
        if selectedIndex != -1:
            manager.runCommand("Update")
        set_roi_and_update_roi_manager(0)
    if keycode == KeyEvent.VK_R:
        selectedIndex = manager.getSelectedIndex()
        if selectedIndex != -1:
            manager.runCommand("Update")
        set_roi_and_update_roi_manager(manager.getCount() - 1)
    if keycode == KeyEvent.VK_G:  # update drawing
        manager.runCommand("Update")
        wafer.manager_to_wafer()
    if keycode == KeyEvent.VK_T:
        wafer.close_mode()
        wafer.start_global_mode()
    if keycode == KeyEvent.VK_H:
        IJ.showMessage("Help for local mode", HELP_MSG_LOCAL)
    if keycode == KeyEvent.VK_G:
        propagate_to_next_section()
    keyEvent.consume()


def handle_key_m_global():
    """Saves overview"""
    manager = wafer.manager
    for roi in manager.iterator():
        annotation_type, section_id, annotation_id = type_id(roi.getName())
        if annotation_type is AnnotationType.SECTION:
            roi.setName(str(section_id))
            roi.setStrokeWidth(8)
        else:
            roi.setName("")
            roi.setStrokeWidth(1)
    IJ.run(
        "Labels...",
        (
            "color=white font="
            + str(int(wafer.image.getWidth() / 400.0))
            + " show use draw bold"
        ),
    )
    flattened = wafer.image.flatten()
    flattened_path = os.path.join(wafer.root, "overview_global.jpg")
    IJ.save(flattened, flattened_path)
    IJ.log("Flattened global image saved to {}".format(flattened_path))
    flattened.close()
    wafer.start_global_mode()


def handle_key_m_local():
    """Saves overview"""
    montageMaker = MontageMaker()
    stack = wafer.image.getStack()
    n_slices = wafer.image.getNSlices()

    n_rows = int(n_slices**0.5)
    n_cols = n_slices // n_rows
    if n_rows * n_cols < n_slices:
        n_rows += 1

    # adjust handle/stroke size depending on image dimensions
    im_w = wafer.image.getWidth()
    montage_factor = (
        1 if im_w < LOCAL_SIZE_STANDARD else LOCAL_SIZE_STANDARD / float(im_w)
    )
    if im_w < LOCAL_SIZE_STANDARD:
        handle_size = 5
        stroke_size = 3
    else:
        handle_size = 5 * intr(im_w / LOCAL_SIZE_STANDARD)
        stroke_size = 3 * im_w / LOCAL_SIZE_STANDARD

    flattened_ims = []
    for id, section_id in enumerate(sorted(wafer.sections.keys())):
        im_p = stack.getProcessor(wafer.serialorder.index(id) + 1).duplicate()
        flattened = ImagePlus("flattened", im_p)

        for roi in wafer.manager.iterator():
            if "-{:04}".format(section_id) in roi.getName():
                cloned_roi = roi.clone()
                cloned_roi.setHandleSize(handle_size)
                cloned_roi.setStrokeWidth(stroke_size)
                flattened.setRoi(cloned_roi)
                flattened = flattened.flatten()
        flattened.setTitle("section-{:04}".format(section_id))
        flattened_ims.append(flattened)
    flattened_stack = ImageStack(
        flattened_ims[0].getWidth(), flattened_ims[0].getHeight()
    )
    for flattened in flattened_ims:
        flattened_stack.addSlice(flattened.getTitle(), flattened.getProcessor())
    montage_stack = ImagePlus("Montage", flattened_stack)
    montage = montageMaker.makeMontage2(
        montage_stack,
        n_rows,
        n_cols,
        montage_factor,
        1,
        montage_stack.getNSlices(),
        1,
        3,
        True,
    )
    flattened_path = os.path.join(wafer.root, "overview_local.jpg")
    IJ.save(montage, flattened_path)
    del flattened_ims
    IJ.log("Flattened local image saved to {}".format(flattened_path))


def handle_key_x_local():
    wafer.remove_current()


def handle_key_p_local():
    """Propagation tool"""
    manager = wafer.manager
    manager.runCommand("Update")  # to update the current ROI
    wafer.manager_to_wafer()
    selected_indexes = manager.getSelectedIndexes()
    if len(selected_indexes) != 1:
        IJ.showMessage(
            "Warning",
            "To use the propagation tool, only one annotation should be selected",
        )
        return
    selected_poly = manager.getRoi(selected_indexes[0])
    poly_name = selected_poly.getName()
    annotation_type, section_id, annotation_id = type_id(poly_name)
    if annotation_type is AnnotationType.SECTION:
        IJ.showMessage(
            "Info",
            "Sections cannot be propagated. Only rois, focus, magnets can be propagated",
        )
        return
    min_section_id = min(wafer.sections)
    max_section_id = max(wafer.sections)

    gd = GenericDialogPlus("Propagation")
    gd.addMessage(
        "This {} is defined in section number {}.\nTo what sections do you want to propagate this {}?".format(
            annotation_type.string, section_id, annotation_type.string
        )
    )
    gd.addStringField(
        "Enter a range or single values separated by commas. "
        + "Range can be start-end (4-7 = 4,5,6,7) or "
        + "start-end-increment (2-11-3 = 2,5,8,11).",
        "{}-{}".format(min_section_id, max_section_id),
    )
    gd.addButton(
        "All sections {}-{}".format(min_section_id, max_section_id),
        ButtonClick(),
    )
    gd.addButton(
        "First half of the sections {}-{}".format(
            min_section_id, intr(max_section_id / 2.0)
        ),
        ButtonClick(),
    )
    gd.addButton(
        "Every second section {}-{}-2".format(min_section_id, max_section_id),
        ButtonClick(),
    )
    gd.showDialog()
    if not gd.wasOKed():
        return
    user_range = gd.getNextString()
    input_indexes = get_indexes_from_user_string(user_range)
    IJ.log("User input indexes from Propagation Dialog: {}".format(input_indexes))
    valid_input_indexes = [i for i in input_indexes if i in wafer.sections]
    if not valid_input_indexes:
        return

    with wafer.set_mode(Mode.GLOBAL):
        if annotation_type is AnnotationType.ROI:
            annotation_points = wafer.rois[section_id][annotation_id].points
        else:
            annotation_points = getattr(wafer, annotation_type.name)[
                annotation_id
            ].points
        for input_index in valid_input_indexes:
            propagated_points = GeometryCalculator.propagate_points(
                wafer.sections[section_id].points,
                annotation_points,
                wafer.sections[input_index].points,
            )
            wafer.add(
                annotation_type,
                GeometryCalculator.points_to_poly(propagated_points),
                (input_index, annotation_id)
                if annotation_type is AnnotationType.ROI
                else input_index,
            )
    wafer.wafer_to_manager()


def propagate_to_next_section():
    """
    In local mode, propagates all annotations of the current section
    to the next serial section
    """

    manager = wafer.manager
    manager.runCommand("Update")  # to update the current ROI
    wafer.manager_to_wafer()

    slice_id = wafer.image.getSlice()
    section_id = sorted(wafer.sections.keys())[wafer.serialorder[slice_id - 1]]
    if slice_id not in wafer.serialorder:
        dlog("Cannot propagate to the next section: there is no next section")
        return
    next_section_id = sorted(wafer.sections.keys())[wafer.serialorder[slice_id]]

    with wafer.set_mode(Mode.GLOBAL):
        for annotation_type in AnnotationType.section_annotations():
            if section_id not in getattr(wafer, annotation_type.name):
                continue
            if annotation_type is AnnotationType.ROI:
                for roi_id, roi in wafer.rois[section_id].iteritems():
                    propagated_points = GeometryCalculator.propagate_points(
                        wafer.sections[section_id].points,
                        roi.points,
                        wafer.sections[next_section_id].points,
                    )
                    wafer.add_roi(
                        GeometryCalculator.points_to_poly(propagated_points),
                        (next_section_id, roi_id),
                    )
            else:
                annotation_points = getattr(wafer, annotation_type.name)[
                    section_id
                ].points
                propagated_points = GeometryCalculator.propagate_points(
                    wafer.sections[section_id].points,
                    annotation_points,
                    wafer.sections[next_section_id].points,
                )
                wafer.add(
                    annotation_type,
                    GeometryCalculator.points_to_poly(propagated_points),
                    next_section_id,
                )
    wafer.wafer_to_manager()
    select_roi_by_name(str(wafer.sections[next_section_id]))


def handle_key_a():
    drawn_roi = wafer.image.getRoi()
    if not drawn_roi:
        IJ.showMessage(
            "Info",
            MSG_DRAWN_ROI_MISSING,
        )
        return
    if drawn_roi.getState() is PolygonRoi.CONSTRUCTING and drawn_roi.size() > 3:
        return

    name_suggestions = []
    if wafer.mode is Mode.LOCAL:
        section_id = wafer.get_section_id_from_local_position()

        if drawn_roi.size() == 2:
            name_suggestions.append("magnet-{:04}".format(section_id))
        else:
            name_suggestions.append("section-{:04}".format(section_id))
            name_suggestions += wafer.suggest_roi_ids(section_id)
        name_suggestions.append("focus-{:04}".format(section_id))
    elif wafer.mode is Mode.GLOBAL:
        closest_sections = wafer.get_closest(
            AnnotationType.SECTION, drawn_roi.getContourCentroid()
        )
        if drawn_roi.size() == 2:
            name_suggestions += [
                "magnet-{:04}".format(section.id_) for section in closest_sections[:3]
            ]
            name_suggestions += [
                "landmark-{:04}".format(id)
                for id in wafer.suggest_annotation_ids(AnnotationType.LANDMARK)
            ]
        else:
            name_suggestions += [
                "section-{:04}".format(id)
                for id in wafer.suggest_annotation_ids(AnnotationType.SECTION)
            ]
            name_suggestions += [
                name_suggestion
                for section in closest_sections[:3]
                for name_suggestion in wafer.suggest_roi_ids(section.id_)
            ]

    annotation_name = annotation_name_validation_dialog(name_suggestions)
    if annotation_name is None:
        return

    points = GeometryCalculator.poly_to_points(drawn_roi)
    # handle cases when the drawn_roi is not closed
    if drawn_roi.getState() == PolygonRoi.CONSTRUCTING:
        if drawn_roi.size() == 3:
            drawn_roi = PolygonRoi(
                [point[0] for point in points[:-1]],
                [point[1] for point in points[:-1]],
                PolygonRoi.POLYLINE,
            )
        elif drawn_roi.size() == 2:
            drawn_roi = PointRoi(*points[0])

    drawn_roi.setName(annotation_name)
    annotation_type, section_id, annotation_id = type_id(annotation_name)
    if wafer.mode is Mode.LOCAL:
        drawn_roi.setHandleSize(annotation_type.handle_size_local)
    else:
        drawn_roi.setHandleSize(annotation_type.handle_size_global)
    if annotation_type is AnnotationType.ROI:
        wafer.add_roi(drawn_roi, (section_id, annotation_id))
    else:
        wafer.add(annotation_type, drawn_roi, annotation_id)
    drawn_roi.setStrokeColor(annotation_type.color)
    wafer.image.killRoi()
    wafer.wafer_to_manager()

    if wafer.mode is Mode.LOCAL:
        # select the drawn_roi
        wafer.manager.select(get_roi_index_by_name(annotation_name))
    else:
        roi_manager_scroll_bottom()
    IJ.log("Annotation {} added".format(annotation_name))


def move_fov(a):
    """Moves field of view of the image"""
    im = wafer.image
    canvas = im.getCanvas()
    r = canvas.getSrcRect()
    # adjust increment depending on zoom level
    increment = intr(40 / float(canvas.getMagnification()))
    xPixelIncrement = 0
    yPixelIncrement = 0
    if a == "right":
        xPixelIncrement = increment
    elif a == "left":
        xPixelIncrement = -increment
    elif a == "down":
        yPixelIncrement = increment
    elif a == "up":
        yPixelIncrement = -increment
    newR = Rectangle(
        min(max(0, r.x + xPixelIncrement), im.getWidth() - r.width),
        min(max(0, r.y + yPixelIncrement), im.getHeight() - r.height),
        r.width,
        r.height,
    )
    canvas.setSourceRect(newR)
    im.updateAndDraw()


def intr(x):
    """Float to int with rounding (instead of int(x) that does a floor)"""
    return int(round(x))


def get_name(text, default_name="", cancel_msg=None):
    gd = GenericDialog(text)
    gd.addStringField(text, default_name)
    gd.showDialog()
    if gd.wasCanceled() and cancel_msg is not None:
        IJ.showMessage(cancel_msg)
        sys.exit()
    return gd.getNextString()


def get_OK(text, window_name="User prompt"):
    gd = GenericDialog(window_name)
    gd.addMessage(text)
    gd.hideCancelButton()
    gd.enableYesNoCancel()
    focus_on_ok(gd)
    gd.showDialog()
    return gd.wasOKed()


def get_indexes_from_user_string(userString):
    """inspired by the substackMaker of ImageJ \n
    https://imagej.nih.gov/ij/developer/api/ij/plugin/SubstackMaker.html
    Enter a range (2-30), a range with increment (2-30-2), or a list (2,5,3)
    """
    userString = userString.replace(" ", "")
    if "," in userString and "." in userString:
        return None
    elif "," in userString:
        splitIndexes = [
            int(splitIndex)
            for splitIndex in userString.split(",")
            if splitIndex.isdigit()
        ]
        if len(splitIndexes) > 0:
            return splitIndexes
    elif "-" in userString:
        splitIndexes = [
            int(splitIndex)
            for splitIndex in userString.split("-")
            if splitIndex.isdigit()
        ]
        if len(splitIndexes) == 2 or len(splitIndexes) == 3:
            splitIndexes[1] = (
                splitIndexes[1] + 1
            )  # inclusive is more natural (2-5 = 2,3,4,5)
            return range(*splitIndexes)
    elif userString.isdigit():
        return [int(userString)]
    return None


def show_dialog(title, subtitle, suggestions, id_default):
    gd = GenericDialog(title)
    gd.addRadioButtonGroup(
        subtitle,
        suggestions,
        len(suggestions),
        1,
        suggestions[id_default],
    )
    focus_on_ok(gd)
    gd.showDialog()
    if gd.wasCanceled():
        return None
    return gd.getRadioButtonGroups()[0].getSelectedCheckbox()


def annotation_name_validation_dialog(name_suggestions):
    checkbox = show_dialog(
        "Validate the name of the annotation",
        "Name of the annotation",
        name_suggestions,
        0,
    )
    if not checkbox:
        IJ.log("No selection made: the section was not added")
        return None
    if checkbox is None:
        return None
    return checkbox.label


def thread_focus_on_OK(button):
    time.sleep(0.1)
    try:
        button.requestFocusInWindow()
    except Exception as e:
        pass


def focus_on_ok(dialog):
    """Trick to preselect the OK button after a dialog has been shown"""
    ok_buttons = [
        button
        for button in list(dialog.getButtons())
        if button and button.getLabel() == "  OK  "
    ]
    try:
        ok_button = ok_buttons[0]
        threading.Thread(target=thread_focus_on_OK, args=[ok_button]).start()
    except Exception as e:
        pass


# ----- RoiManager functions ----- #
def get_roi_manager():
    manager = RoiManager.getInstance()
    if manager is None:
        manager = RoiManager()
    return manager


def init_manager():
    # get, place, and reset the ROIManager
    manager = get_roi_manager()
    manager.setTitle("Annotations")
    manager.reset()
    manager.setSize(
        250, intr(0.95 * IJ.getScreenSize().height)
    )  # 280 so that the title is not cut
    manager.setLocation(IJ.getScreenSize().width - manager.getSize().width, 0)
    return manager


def roi_manager_scroll_bottom():
    manager = get_roi_manager()
    scrollPane = [
        component
        for component in manager.getComponents()
        if "Scroll" in str(type(component))
    ][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barMax = scrollBar.getMaximum()
    scrollBar.setValue(intr(1.5 * barMax))


def move_roi_manager_selection(n):
    manager = get_roi_manager()
    if manager.getCount() == 0:
        return
    selected_index = manager.getSelectedIndex()
    if selected_index == -1:
        roi_id = get_roi_index_from_current_slice()
        if roi_id is not None:
            set_roi_and_update_roi_manager(roi_id)
    else:
        manager.runCommand("Update")
        if n < 0:
            set_roi_and_update_roi_manager(max(0, selected_index + n))
        elif n > 0:
            set_roi_and_update_roi_manager(
                min(manager.getCount() - 1, selected_index + n)
            )


def delete_selected_roi():
    wafer.manager.runCommand("Delete")


def delete_roi_by_index(index):
    wafer.manager.select(index)
    wafer.manager.runCommand("Delete")


def select_roi_by_name(roi_name):
    manager = get_roi_manager()
    roi_index = [roi.getName() for roi in manager.getRoisAsArray()].index(roi_name)
    manager.select(roi_index)


def set_roi_and_update_roi_manager(roi_index, select=True):
    """Keeps the selected annotation in the middle of the manager display"""
    manager = get_roi_manager()
    nRois = manager.getCount()
    scrollPane = [
        component
        for component in manager.getComponents()
        if "Scroll" in str(type(component))
    ][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barMax = scrollBar.getMaximum()
    barWindow = scrollBar.getVisibleAmount()
    roisPerBar = nRois / float(barMax)
    roisPerWindow = barWindow * roisPerBar
    scrollValue = int((roi_index - roisPerWindow / 2.0) / float(roisPerBar))
    scrollValue = max(0, scrollValue)
    scrollValue = min(scrollValue, barMax)
    if select:
        manager.select(roi_index)
    # if nRois>roisPerWindow:
    # scrollBar.setValue(scrollValue)
    scrollBar.setValue(scrollValue)


def get_roi_index_from_current_slice():
    manager = get_roi_manager()
    slice_id = wafer.image.getSlice()
    # find the rois that are assigned to that slice
    for roi in manager.iterator():
        if roi.getPosition() == slice_id:
            break  # TODO: confirm that it is finding the section first
    roi_index = manager.getRoiIndex(roi)
    if roi_index == -1:
        if manager.getCount() > 0:
            return 0
        return None
    return roi_index


def get_roi_index_by_name(name):
    manager = wafer.manager
    try:
        index = [manager.getName(i) for i in range(manager.getCount())].index(name)
        return index
    except ValueError as e:
        return None


def toggle_fill(annotation_type):
    manager = wafer.manager
    for roi in manager.iterator():
        if annotation_type.string in roi.getName():
            if roi.getFillColor():
                roi.setFillColor(None)
            else:
                roi.setFillColor(annotation_type.color)
    manager.runCommand("Show All without labels")


def toggle_labels():
    if wafer.manager.getDrawLabels():
        wafer.manager.runCommand("Show All without labels")
    else:
        wafer.manager.runCommand("UseNames", "true")
        wafer.manager.runCommand("Show All with labels")


def type_id(
    name,
    delimiter_0="-",
    delimiter_1=".",
):
    """Returns annotation_type, section_id, annotation_id given an annotation name

    type_id(section-0012) -> AnnotationType.SECTION, 12
    type_id(roi-0019.01) -> AnnotationType.ROI, 1901
    type_id(roi-002.01) -> AnnotationType.ROI, 201
    type_id(roi.0019.01, delimiter_0=".") -> AnnotationType.ROI, 1901
    """
    parts = name.split(delimiter_0)
    ids = [int(x) for x in delimiter_0.join(parts[1:]).split(delimiter_1)]
    annotation_type = [t for t in AnnotationType.all() if t.string in parts[0]][0]
    if annotation_type is AnnotationType.ROI:
        return annotation_type, ids[0], ids[1]
    return annotation_type, ids[0], ids[0]


def suggest_ids(ids):
    """[1,3,4] -> [0,2,5]"""
    if not ids:
        return [0]
    return sorted(set(range(max(ids) + 2)) - set(ids))


# ----- End RoiManager functions ----- #
# ----- HTML functions for help message ----- #
def tag(text, tag):
    return "<{}>{}</{}>".format(tag, text, tag)


def print_list(*elements):
    a = "<ul>"
    for s in elements:
        a += "<li>{}</li>".format(s)
    a += "</ul>"
    return a


# ----- End HTML functions for help message ----- #


def pairwise(iterable):
    """from https://docs.python.org/3/library/itertools.html#itertools.pairwise
    'ABCD' --> AB BC CD"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == "__main__":
    HELP_MSG_GLOBAL = (
        "<html><br>[a] = Press the key a<br><br>"
        + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
        + tag("Navigation", "h3")
        + print_list(
            "Zoom in/out"
            + print_list(
                "[Ctrl] + mouse wheel",
                "[+] / [-]",
            ),
            "Move up/down" + print_list("mouse wheel", "[&uarr] / [&darr]"),
            "Move left/right"
            + print_list("[Shift] + Mouse wheel", "[&larr] / [&rarr]"),
            "Toggle"
            + print_list(
                "filling of sections   [1]",
                "filling of rois   [2]",
                "filling of focus   [3]",
                "labels [0] (number 0)",
            ),
        )
        + tag("Action", "h3")
        + print_list(
            "Mouse click"
            + print_list(
                "Left - draws point",
                "Right - joins first and last points to close the current polygon",
            ),
            "[h] opens this help message",
            "[escape] stops current drawing",
            "[a] adds an annotation that you have drawn",
            "[t] toggles to local mode",
            "[q] quits (everything will be saved)",
            "[s] saves (happens automatically when toggling [t] and quitting [q])",
            "[m] exports a summary image of global mode",
            "[n] renumbers the sections to have continuous numbers without gaps (0,1,3,4,6 --> 0,1,2,3,4)",
            (
                "[o]   (letter o) computes the section order that minimizes the travel of the microscope stage"
                " (not the same as the serial sectioning order of the sections)."
                ' Saves the order in the .magc file in the field "stageorder"'
            ),
            (
                "[j] computes the serial order based on the roi defined in the first section."
                " Updates the section positions."
                'Stores the serial order in the .magc file in the field "serialorder".'
                " We recommend to save beforehand a copy of the .magc file outside of the directory"
            ),
        )
        + "<br><br><br></html>"
    )
    HELP_MSG_LOCAL = (
        "<html><br><br><br>[a] = Press the key a<br><br>"
        + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
        + tag("Navigation", "h3")
        + print_list(
            "Navigate annotations up/down" + print_list("[d]/[f]", "Mouse wheel"),
            "Navigate 10 annotations up/down"
            + print_list("[c]/[v]", "[Ctrl] + Mouse wheel"),
            "[e]/[r] navigate to first/last annotation",
            "If you lose the current annotation (by clicking outside of the annotation), then press [d],[f] or use the mouse wheel to make the annotation appear again.",
        )
        + tag("Action", "h3")
        + print_list(
            "[h] opens this help message",
            "[a] creates/modifies an annotation",
            "[t] toggles to global mode",
            "[p] propagates the current annotation to sections defined in the dialog. Section and landmark annotations cannot be propagated",
            "[g] propagates all annotations of the current section to the next serial section and moves to the next serial section",
            "[q] quits. Everything will be saved",
            "[s] saves to file (happens already automatically when toggling [t] or quitting [q])",
            "[m] exports a summary montage",
            (
                "[o]   (letter o) computes the section order that minimizes the travel of the microscope stage"
                " (not the same as the serial sectioning order of the sections)."
                ' Saves the order in the .magc file in the field "stageorder"'
            ),
        )
        + "<br><br><br></html>"
    )

    initial_ij_key_listeners = IJ.getInstance().getKeyListeners()
    wafer = Wafer()
    wafer.start_global_mode()
