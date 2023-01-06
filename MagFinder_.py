from __future__ import with_statement

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
DISPLAY_FACTOR = 1.2
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


def dlog(x):
    """Double log to print and dlog"""
    IJ.log(x)
    print(x)


def intr(x):
    """Float to int with rounding (instead of int(x) that does a floor)"""
    return int(round(x))


def pairwise(iterable):
    """from https://docs.python.org/3/library/itertools.html#itertools.pairwise
    'ABCD' --> AB BC CD"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_square_row_col(n):
    """Returns a grid close to being square with n_rows*n_cols > n"""
    n_rows = int(n**0.5)
    n_cols = n // n_rows
    if n_rows * n_cols < n:
        n_rows += 1
    return n_rows, n_cols


class Direction(object):
    """Enum for directions"""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class Mode(object):
    """Enum for the display modes"""

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
    ROI = AnnotationTypeDef("rois", "roi", Color.yellow, 15, 2)
    FOCUS = AnnotationTypeDef("focus", "focus", Color.green, 15, 15)
    MAGNET = AnnotationTypeDef("magnets", "magnet", Color.green, 15, 15)
    LANDMARK = AnnotationTypeDef("landmarks", "landmark", Color.yellow, 15, 15)

    @classmethod
    def all(cls):
        """Returns all annotation types"""
        return [cls.SECTION, cls.ROI, cls.FOCUS, cls.MAGNET, cls.LANDMARK]

    @classmethod
    def all_but_landmark(cls):
        """Returns all annotation types except landmarks"""
        return [cls.SECTION, cls.ROI, cls.FOCUS, cls.MAGNET]

    @classmethod
    def section_annotations(cls):
        """Returns all annotation types except landmarks"""
        return [cls.ROI, cls.FOCUS, cls.MAGNET]


class Wafer(object):
    def __init__(self, magc_path=None):
        self.mode = Mode.GLOBAL
        self.root, self.image_path = self.init_image_path(magc_path)
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
        self.serial_order = []
        """the serial order is the order in which the sections have been cut"""
        self.stage_order = []
        """the stage_order is the order that minimizes microscope stage travel
        to image one section after the other"""
        self.GC = GeometryCalculator
        self.file_to_wafer()
        IJ.setTool("polygon")

    def __len__(self):
        """Returns the number of sections"""
        if not hasattr(self, "sections"):
            return 0
        return len(self.sections)

    # TODO this image vs img does not look optimal
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

    @property
    def landmarks_xy(self):
        return [self.landmarks[key].centroid for key in sorted(self.landmarks)]

    @contextmanager
    def set_mode(self, mode):
        old_mode = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = old_mode

    def set_listeners(self):
        """Sets key and mouse wheel listeners"""
        add_key_listener_everywhere(KeyListener(self))
        add_mouse_wheel_listener_everywhere(MouseWheelListener(self))

    def set_global_mode(self):
        """useful when wafer accessed from another module"""
        self.mode = Mode.GLOBAL

    def set_local_mode(self):
        self.mode = Mode.LOCAL

    def init_image_path(self, magc_path=None):
        """
        Finds the image used for navigation in magfinder.
        It is the image with the smallest size in the directory
        that does not contain "overview" in its name.
        """
        if magc_path is not None:
            root = os.path.dirname(magc_path)
        else:
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
                "There is no image (.tif, .png, .jpg, .jpeg, .tiff) in the experiment folder you selected."
                "\nAdd an image and start again the plugin"
                "\n(Disregard this message if you are doing a wafer transfer and know what you are doing)",
            )
            # sys.exit()
            return root, None
        return root, wafer_im_path

    def init_magc_path(self):
        """Loads existing .magc file or creates a new one if does not exist"""
        magc_paths = [
            os.path.join(self.root, filename)
            for filename in os.listdir(self.root)
            if filename.endswith(".magc")
        ]
        if not magc_paths:
            dlog("No .magc file found. Creating an empty one")
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
        config = ConfigParser.ConfigParser()
        with open(self.magc_path, "rb") as configfile:
            config.readfp(configfile)
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
            elif header in ["serial_order", "stage_order"]:
                if config.get(header, header) != "[]":
                    setattr(
                        self,
                        header,
                        [int(x) for x in config.get(header, header).split(",")],
                    )
        if not self.serial_order:
            self.serial_order = sorted(self.sections.keys())
        if not self.stage_order:
            self.stage_order = sorted(self.sections.keys())
        dlog(
            (
                "File successfully read with \n{} sections \n{} rois \n{} focus"
                + "\n{} magnets \n{} landmarks"
            ).format(
                len(self),
                len(self.rois),
                len(self.focus),
                len(self.magnets),
                len(self.landmarks),
            )
        )

    def manager_to_wafer(self):
        """
        Populates the wafer from the roi manager
        Typically called after the user interacted with the UI
        """
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

    def wafer_to_manager(self):
        """Draws all rois from the wafer instance into the manager"""
        self.manager.reset()
        if self.mode is Mode.GLOBAL:
            self.wafer_to_manager_global()
        else:
            self.wafer_to_manager_local()

    def wafer_to_manager_global(self):
        for landmark in self.landmarks.values():
            if not (
                0 < landmark.centroid[0] < self.image.getWidth()
                and 0 < landmark.centroid[1] < self.image.getHeight()
            ):
                # do not draw landmark if it is not visible,
                # otherwise it creates a weird offset with negative coordinates
                continue
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
        for id_order, section_id in enumerate(self.serial_order):
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
                    local_poly.setPosition(0, id_order + 1, 0)
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
                local_poly.setPosition(0, id_order + 1, 0)
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

    def save(self):
        """Saves the wafer annotations to the .magc file"""
        dlog("Saving ...")
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
        for order_name in ["serial_order", "stage_order"]:
            config.add_section(order_name)
            order = getattr(self, order_name)
            if not order:
                dlog("Debug: this case should not happen".center(100, "-"))
                config.set(order_name, order_name, "[]")
            else:
                config.set(order_name, order_name, ",".join([str(x) for x in order]))

        with open(self.magc_path, "w") as configfile:
            config.write(configfile)
        dlog("Saved to {}".format(self.magc_path))
        self.save_csv()

    def save_csv(self):
        # TODO currently broken with multirois
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
                                self.stage_order[id]
                                if len(self.stage_order) > id
                                else "",
                                self.serial_order[id]
                                if len(self.serial_order) > id
                                else "",
                            ]
                        ],
                    )
                )
                # unusual case: if there are more landmarks than sections
                for i in range(len(self), len(self.landmarks)):
                    f.write(
                        ",,,,,,,,,{},{},,".format(
                            self.landmarks[i].centroid[0], self.landmarks[i].centroid[1]
                        )
                    )
                f.write("\n")
        dlog("Annotations saved to {}".format(csv_path))

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
        dlog("Starting local mode ...")
        self.mode = Mode.LOCAL
        self.compute_transforms()
        self.create_local_stack()
        self.wafer_to_manager()
        self.set_listeners()
        self.manager.runCommand("UseNames", "false")
        self.manager.runCommand("Show None")
        set_roi_and_update_roi_manager(0)  # select first ROI
        self.arrange_windows()

    def compute_transforms(self):
        """
        1.self.transforms[section_key] transforms the global wafer image
        to an image in which the section section_key is centered at 0
        and has an angle of 0 degrees
        2.the poly_transforms are almost like the self.transforms except that
        they contain an offset due to the fact that an ImagePlus is displayed
        with their top-left corner at 0,0 and not at -w/2,-h/2
        """
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

    def create_local_stack(self):
        """Creates the local stack with imglib2 framework"""
        display_params = (
            [-intr(0.5 * v) for v in self.local_display_size],
            [intr(0.5 * v) for v in self.local_display_size],
        )
        imgs = [
            Views.interval(
                RV.transform(
                    Views.interpolate(
                        Views.extendZero(self.img_global),
                        # NLinearInterpolatorFactory()
                        NearestNeighborInterpolatorFactory(),
                    ),
                    self.transforms[o],
                ),
                display_params[0],
                display_params[1],
            )
            for o in self.serial_order
        ]
        self.img_local = Views.permute(
            Views.addDimension(Views.stack(imgs), 0, 0), 3, 2
        )
        IL.show(self.img)
        self.image_local = IJ.getImage()

    def add(self, annotation_type, poly, annotation_id):
        """
        Adds an annotation to the wafer and returns it
        annotation_id is a tuple for rois (section_id, roi_id)
        and an int otherwise
        """

        if annotation_type is AnnotationType.ROI:
            section_id = annotation_id[0]
        else:
            section_id = annotation_id
        if (
            annotation_type is AnnotationType.SECTION
            and annotation_id not in self.serial_order
            # must use serial order, not self.sections
            # because some workflows must handle serial_order changes themselves
            #  (e.g. renumber sections)
        ):
            # appends to serial order only if new section
            self.serial_order.append(section_id)
            self.stage_order.append(section_id)
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
                else:
                    del getattr(self, annotation_type.name)[annotation_id]
                # select the section
                self.manager.select(
                    get_roi_index_from_name(str(self.sections[section_id]))
                )
        elif annotation_type is AnnotationType.SECTION:
            # deleting a sections also deletes the linked annotations (roi(s),focus,magnet)
            section_id_manager = get_roi_index_from_name(str(self.sections[section_id]))
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
            self.image.killRoi()
            # delete linked annotations in manager and in wafer
            for linked_annotation in linked_annotations:
                # delete in roimanager
                index = get_roi_index_from_name(str(linked_annotation))
                delete_roi_by_index(index)

                # delete in wafer
                annotation_type, section_id, annotation_id = type_id(
                    str(linked_annotation)
                )
                if annotation_type is AnnotationType.ROI:
                    del self.rois[section_id][annotation_id]
                else:
                    del getattr(self, linked_annotation.type_.name)[section_id]
            # delete section in manager
            section_id_manager = get_roi_index_from_name(str(self.sections[section_id]))
            delete_roi_by_index(section_id_manager)
            # update the orders
            del self.serial_order[self.serial_order.index(section_id)]
            del self.stage_order[self.stage_order.index(section_id)]

            del self.sections[section_id]
            del self.transforms[section_id]
            del self.poly_transforms[section_id]
            del self.poly_transforms_inverse[section_id]

            self.image.close()
            self.manager.reset()

            if len(self) == 0:
                self.start_global_mode()
            else:
                self.start_local_mode()
                # select the next section
                if section_id_manager < self.manager.getCount():
                    set_roi_and_update_roi_manager(section_id_manager)
                else:
                    set_roi_and_update_roi_manager(self.manager.getCount() - 1)

    def init_images_global(self):
        if self.image_path is None:
            dlog("Warning: images not properly initialized")
            return
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

    def suggest_annotation_ids(self, annotation_type):
        item_ids = [item.id_ for item in getattr(self, annotation_type.name).values()]
        return suggest_ids(item_ids)

    def get_closest(self, annotation_type, point):
        """Get the annotations of a certain type closest to a given point"""
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
        for section in self.sections.values()[: min(5, len(self))]:
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
        sorted_section_keys = sorted(self.sections)
        center_points = [
            pFloat(*wafer.sections[section_key].centroid)
            for section_key in sorted(sorted_section_keys)
        ]
        # fill distance matrix
        distances = TSPSolver.init_mat(len(self), initValue=999999)
        for a, b in itertools.combinations_with_replacement(range(len(self)), 2):
            distances[a][b] = distances[b][a] = center_points[a].distance(
                center_points[b]
            )
        self.stage_order = [
            sorted_section_keys[o] for o in self.tsp_solver.compute_tsp_order(distances)
        ]

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
        if self.mode is Mode.LOCAL:
            dlog("Closing local mode. Section renumbering will be done in global mode")
            self.close_mode()
            self.start_global_mode()
        for new_key, key in enumerate(sorted(self.sections)):
            if new_key == key:
                continue
            # update orders
            id_serial = self.serial_order.index(key)
            self.serial_order[id_serial] = new_key
            id_stage = self.stage_order.index(key)
            self.stage_order[id_stage] = new_key
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
            for subroi_id, subroi in self.rois[key].iteritems():
                self.add_roi(subroi.poly, (new_key, subroi_id))
                del self.rois[key][subroi_id]
        self.clear_transforms()
        self.compute_transforms()
        self.wafer_to_manager()
        dlog("{} sections have been renumbered".format(len(self)))

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

    def export_transforms(self):
        IJ.log("Saving transforms...")
        if not self.transforms:
            self.compute_transforms()
        folder_transforms = mkdir_p(os.path.join(self.root, "section_transforms"))
        for id_slab, transform in sorted(self.transforms.iteritems()):
            path_transform = os.path.join(
                folder_transforms, "transform_{:04}.txt".format(id_slab)
            )
            with open(path_transform, "w") as f:
                f.write(transform.toString())
        IJ.log("Saved transforms.")

    def transfer_from_source_wafer(self):
        """
        Imports annotations from a source wafer .magc file.
        The same ficucials must be defined in the two wafers.
        The fiducials are used to map all annotations from the source wafer
        into this current target wafer.
        Only available in global mode.
        """
        wafer_target = self
        path_source = get_path(
            "Select .magc file from which annotations should be imported"
        )
        if path_source is None:
            return
        wafer_source = Wafer(magc_path=path_source)

        aff = GeometryCalculator.affine_t(
            [l[0] for l in wafer_source.landmarks_xy],
            [l[1] for l in wafer_source.landmarks_xy],
            [l[0] for l in wafer_target.landmarks_xy],
            [l[1] for l in wafer_target.landmarks_xy],
        )
        wafer_target.clear_annotations()
        wafer_target.clear_transforms()
        for key_section in sorted(wafer_source.sections):
            for annotation_type in AnnotationType.all_but_landmark():
                if not (
                    hasattr(wafer_source, annotation_type.name)
                    and key_section in getattr(wafer_source, annotation_type.name)
                ):
                    continue
                if annotation_type is AnnotationType.ROI:
                    rois = wafer_source.rois[key_section]
                    if not rois:
                        continue
                    for key_roi, roi in rois.iteritems():
                        wafer_target.add(
                            annotation_type,
                            GeometryCalculator.points_to_poly(
                                GeometryCalculator.xy_to_points(
                                    *GeometryCalculator.apply_affine_t(
                                        [p[0] for p in roi.points],
                                        [p[1] for p in roi.points],
                                        aff,
                                    )
                                )
                            ),
                            (key_section, key_roi),
                        )
                else:
                    annotation = getattr(wafer_source, annotation_type.name)[
                        key_section
                    ]
                    wafer_target.add(
                        annotation_type,
                        GeometryCalculator.points_to_poly(
                            GeometryCalculator.xy_to_points(
                                *GeometryCalculator.apply_affine_t(
                                    [p[0] for p in annotation.points],
                                    [p[1] for p in annotation.points],
                                    aff,
                                )
                            )
                        ),
                        key_section,
                    )
        wafer_target.stage_order = wafer_source.stage_order
        wafer_target.serial_order = wafer_source.serial_order
        wafer_target.compute_transforms()
        wafer_target.wafer_to_manager()
        dlog("Completed transfer from {} to {} ...".format(wafer_source, wafer_target))


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
        """Returns bounding box area. getStatistics.pixelcount was too slow"""
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
            return
        return self.poly.getFloatAngle(
            self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1]
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
                        dlog("Failed to download cygwin1.dll due to {}".format(e))
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
        dlog("Downloading TSP solver from " + str(url))
        gz_path = os.path.join(IJ.getDirectory("plugins"), "temp.gz")
        try:
            FileUtils.copyURLToFile(URL(url), File(gz_path))
            gis = GZIPInputStream(FileInputStream(gz_path))
            Files.copy(gis, Paths.get(target_path))
            gis.close()
            os.remove(gz_path)
        except (Exception, java_exception) as e:
            dlog("Failed to download from " + str(url) + " due to " + str(e))

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
        # dlog('TSP solving command ' + str(command))
        Runtime.getRuntime().exec(command)

        while not os.path.isfile(solution_path):
            time.sleep(1)
            dlog(
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
            dlog(
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
            dlog(
                "Could not compute the stage-movement-minimizing order"
                " because the solver or cygwin1.dll are missing"
            )
            return
        try:
            order = self.order_from_mat(pairwise_costs, self.root, solver_path)
            dlog("The optimal order is: {}".format(order))
        except (Exception, java_exception) as e:
            dlog("The order could not be computed: {}".format(e))
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


class MouseWheelListener(MouseAdapter):
    def __init__(self, wafer):
        self.wafer = wafer

    def mouseWheelMoved(self, mouseWheelEvent):
        if self.wafer.mode is Mode.GLOBAL:
            self.handle_mouse_wheel_global(mouseWheelEvent)
        else:
            self.handle_mouse_wheel_local(mouseWheelEvent)

    def handle_mouse_wheel_global(self, mouseWheelEvent):
        mouseWheelEvent.consume()
        if mouseWheelEvent.isShiftDown():
            if mouseWheelEvent.getWheelRotation() == 1:
                move_fov(Direction.RIGHT, self.wafer)
            else:
                move_fov(Direction.LEFT, self.wafer)
        elif not mouseWheelEvent.isShiftDown() and not mouseWheelEvent.isControlDown():
            if mouseWheelEvent.getWheelRotation() == 1:
                move_fov(Direction.DOWN, self.wafer)
            else:
                move_fov(Direction.UP, self.wafer)
        elif mouseWheelEvent.isControlDown():
            if mouseWheelEvent.getWheelRotation() == 1:
                IJ.run("Out [-]")
            elif mouseWheelEvent.getWheelRotation() == -1:
                IJ.run("In [+]")

    @staticmethod
    def handle_mouse_wheel_local(mouseWheelEvent):
        mouseWheelEvent.consume()
        if mouseWheelEvent.isControlDown():
            move_roi_manager_selection(10 * mouseWheelEvent.getWheelRotation())
        else:
            move_roi_manager_selection(mouseWheelEvent.getWheelRotation())


class KeyListener(KeyAdapter):
    def __init__(self, wafer):
        self.wafer = wafer
        self.manager = get_roi_manager()  # TODO does it survive a roi manager reset?

    def keyPressed(self, event):
        keycode = event.getKeyCode()
        event.consume()
        if event.getKeyCode() == KeyEvent.VK_J and self.wafer.mode is Mode.GLOBAL:
            MagReorderer.MagReorderer(self.wafer).reorder()
        elif keycode == KeyEvent.VK_S:
            self.wafer.save()
        elif keycode == KeyEvent.VK_Q:  # terminate and save
            self.wafer.manager_to_wafer()  # will be repeated in close_mode but it's OK
            self.wafer.compute_stage_order()
            self.wafer.close_mode()
            self.manager.close()
        elif keycode == KeyEvent.VK_O:
            self.wafer.manager_to_wafer()
            self.wafer.compute_stage_order()
            self.wafer.save()
        elif keycode == KeyEvent.VK_L:
            self.wafer.export_transforms()
        if keycode == KeyEvent.VK_H:
            IJ.showMessage(
                "Help for {} mode".format(self.wafer.mode),
                HELP_MSG_GLOBAL if self.wafer.mode is Mode.GLOBAL else HELP_MSG_LOCAL,
            )
        elif self.wafer.mode is Mode.GLOBAL:
            self.handle_key_global(event)
        elif self.wafer.mode is Mode.LOCAL:
            self.handle_key_local(event)

    def handle_key_global(self, keyEvent):
        keycode = keyEvent.getKeyCode()
        if keycode == KeyEvent.VK_A:
            self.handle_key_a()
        if keycode == KeyEvent.VK_N:
            self.wafer.renumber_sections()
        if keycode == KeyEvent.VK_T:
            if self.wafer.sections:
                self.wafer.close_mode()
                self.wafer.start_local_mode()
            else:
                IJ.showMessage(
                    "Cannot toggle to local mode because there are no sections defined."
                )
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
            move_fov(Direction.UP, self.wafer)
        if keycode == KeyEvent.VK_DOWN:
            move_fov(Direction.DOWN, self.wafer)
        if keycode == KeyEvent.VK_RIGHT:
            move_fov(Direction.RIGHT, self.wafer)
        if keycode == KeyEvent.VK_LEFT:
            move_fov(Direction.LEFT, self.wafer)
        if keycode == KeyEvent.VK_M:
            self.handle_key_m_global()
        if keycode == KeyEvent.VK_K:
            self.wafer.transfer_from_source_wafer()

    def handle_key_m_global(self):
        """Saves overview"""
        for roi in self.manager.iterator():
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
                + str(int(self.wafer.image.getWidth() / 400.0))
                + " show use draw bold"
            ),
        )
        flattened = self.wafer.image.flatten()
        flattened_path = os.path.join(self.wafer.root, "overview_global.jpg")
        IJ.save(flattened, flattened_path)
        dlog("Flattened global image saved to {}".format(flattened_path))
        flattened.close()
        self.wafer.start_global_mode()

    def handle_key_local(self, keyEvent):
        keycode = keyEvent.getKeyCode()
        if keycode == KeyEvent.VK_A:
            self.handle_key_a()
        if keycode == KeyEvent.VK_X:
            self.handle_key_x_local()
        if keycode == KeyEvent.VK_P:
            self.handle_key_p_local()
        if keycode == KeyEvent.VK_M:
            self.handle_key_m_local()
        if keycode == KeyEvent.VK_D:
            move_roi_manager_selection(-1)
        if keycode == KeyEvent.VK_F:
            move_roi_manager_selection(1)
        if keycode == KeyEvent.VK_C:
            move_roi_manager_selection(-10)
        if keycode == KeyEvent.VK_V:
            move_roi_manager_selection(10)
        if keycode in (KeyEvent.VK_E, KeyEvent.VK_R):
            self.handle_keys_e_r(keycode)
        if keycode == KeyEvent.VK_T:
            self.wafer.close_mode()
            self.wafer.start_global_mode()
        if keycode == KeyEvent.VK_G:
            self.propagate_to_next_section()
        keyEvent.consume()

    def handle_keys_e_r(self, keycode):
        selectedIndex = self.manager.getSelectedIndex()
        if selectedIndex != -1:
            self.manager.runCommand("Update")
        set_roi_and_update_roi_manager(
            0 if keycode == KeyEvent.E else self.manager.getCount() - 1
        )

    def handle_key_a(self):
        drawn_roi = self.wafer.image.getRoi()
        if not drawn_roi:
            IJ.showMessage("Info", MSG_DRAWN_ROI_MISSING)
            return
        if drawn_roi.getState() is PolygonRoi.CONSTRUCTING and drawn_roi.size() > 3:
            return

        name_suggestions = []
        if self.wafer.mode is Mode.LOCAL:
            section_id = self.wafer.serial_order[self.wafer.image.getSlice() - 1]
            if drawn_roi.size() == 2:
                name_suggestions.append("magnet-{:04}".format(section_id))
            else:
                name_suggestions.append("section-{:04}".format(section_id))
                name_suggestions += self.wafer.suggest_roi_ids(section_id)
            name_suggestions.append("focus-{:04}".format(section_id))
        elif self.wafer.mode is Mode.GLOBAL:
            closest_sections = self.wafer.get_closest(
                AnnotationType.SECTION, drawn_roi.getContourCentroid()
            )
            if drawn_roi.size() == 2:
                name_suggestions += [
                    "magnet-{:04}".format(section.id_)
                    for section in closest_sections[:3]
                ]
                name_suggestions += [
                    "landmark-{:04}".format(id)
                    for id in self.wafer.suggest_annotation_ids(AnnotationType.LANDMARK)
                ]
            else:
                name_suggestions += [
                    "section-{:04}".format(id)
                    for id in self.wafer.suggest_annotation_ids(AnnotationType.SECTION)
                ]
                name_suggestions += [
                    name_suggestion
                    for section in closest_sections[:3]
                    for name_suggestion in self.wafer.suggest_roi_ids(section.id_)
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
        if self.wafer.mode is Mode.LOCAL:
            drawn_roi.setHandleSize(annotation_type.handle_size_local)
        else:
            drawn_roi.setHandleSize(annotation_type.handle_size_global)
        if annotation_type is AnnotationType.ROI:
            self.wafer.add_roi(drawn_roi, (section_id, annotation_id))
        else:
            self.wafer.add(annotation_type, drawn_roi, annotation_id)
        drawn_roi.setStrokeColor(annotation_type.color)
        self.wafer.image.killRoi()
        self.wafer.wafer_to_manager()

        if self.wafer.mode is Mode.LOCAL:
            # select the drawn_roi
            self.manager.select(get_roi_index_from_name(annotation_name))
        else:
            roi_manager_scroll_bottom()
        dlog("Annotation {} added".format(annotation_name))

    def handle_key_m_local(self):
        """Saves overview"""
        montageMaker = MontageMaker()
        stack = self.wafer.image.getStack()
        n_slices = self.wafer.image.getNSlices()
        n_rows, n_cols = get_square_row_col(n_slices)

        # adjust handle/stroke size depending on image dimensions
        im_w = self.wafer.image.getWidth()
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
        for id_serial, section_id in enumerate(self.wafer.serial_order):
            im_p = stack.getProcessor(id_serial + 1).duplicate()
            flattened = ImagePlus("flattened", im_p)

            for roi in self.manager.iterator():
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
        im_flattened_stack = ImagePlus("flattened_stack", flattened_stack)
        path_flattened_stack = os.path.join(self.wafer.root, "annotated_stack.tif")
        IJ.save(im_flattened_stack, path_flattened_stack)
        montage = montageMaker.makeMontage2(
            im_flattened_stack,
            n_rows,
            n_cols,
            montage_factor,
            1,
            im_flattened_stack.getNSlices(),
            1,
            3,
            True,
        )
        path_overview_local = os.path.join(self.wafer.root, "overview_local.jpg")
        IJ.save(montage, path_overview_local)
        del flattened_ims
        montage.close()
        im_flattened_stack.close()
        dlog("Flattened local image saved to {}".format(path_overview_local))

    def handle_key_x_local(self):
        self.wafer.remove_current()

    def handle_key_p_local(self):
        """Propagation tool"""
        self.manager.runCommand("Update")  # update the current ROI
        self.wafer.manager_to_wafer()
        selected_indexes = self.manager.getSelectedIndexes()
        if len(selected_indexes) != 1:
            IJ.showMessage(
                "Warning", "Select only one annotation to use the propagation tool"
            )
            return
        selected_poly = self.manager.getRoi(selected_indexes[0])
        poly_name = selected_poly.getName()
        annotation_type, section_id, annotation_id = type_id(poly_name)
        if annotation_type is AnnotationType.SECTION:
            IJ.showMessage(
                "Info",
                "Sections cannot be propagated. Only rois, focus, magnets can be propagated",
            )
            return
        min_section_id = min(self.wafer.sections)
        max_section_id = max(self.wafer.sections)

        gd = GenericDialogPlus("Propagation")
        gd.addMessage(
            (
                "This {} is defined in section #{}."
                "\nTo what sections do you want to propagate this {}?"
            ).format(annotation_type.string, section_id, annotation_type.string)
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
        dlog("User input indexes from Propagation Dialog: {}".format(input_indexes))
        valid_input_indexes = [i for i in input_indexes if i in self.wafer.sections]
        if not valid_input_indexes:
            return

        with self.wafer.set_mode(Mode.GLOBAL):
            if annotation_type is AnnotationType.ROI:
                annotation_points = self.wafer.rois[section_id][annotation_id].points
            else:
                annotation_points = getattr(self.wafer, annotation_type.name)[
                    annotation_id
                ].points
            for input_index in valid_input_indexes:
                propagated_points = GeometryCalculator.propagate_points(
                    self.wafer.sections[section_id].points,
                    annotation_points,
                    self.wafer.sections[input_index].points,
                )
                self.wafer.add(
                    annotation_type,
                    GeometryCalculator.points_to_poly(propagated_points),
                    (input_index, annotation_id)
                    if annotation_type is AnnotationType.ROI
                    else input_index,
                )
        self.wafer.wafer_to_manager()

    def propagate_to_next_section(self):
        """
        In local mode, propagates all annotations of the current section
        to the next serial section
        """
        self.manager.runCommand("Update")
        self.wafer.manager_to_wafer()

        slice_id = self.wafer.image.getSlice()
        section_id = self.wafer.serial_order[slice_id - 1]
        try:
            next_section_id = self.wafer.serial_order[slice_id]
        except IndexError:
            dlog("Cannot propagate to the next section: there is no next section")
            return

        with self.wafer.set_mode(Mode.GLOBAL):
            for annotation_type in AnnotationType.section_annotations():
                if section_id not in getattr(self.wafer, annotation_type.name):
                    continue
                if annotation_type is AnnotationType.ROI:
                    for roi_id, roi in self.wafer.rois[section_id].iteritems():
                        propagated_points = GeometryCalculator.propagate_points(
                            self.wafer.sections[section_id].points,
                            roi.points,
                            self.wafer.sections[next_section_id].points,
                        )
                        self.wafer.add_roi(
                            GeometryCalculator.points_to_poly(propagated_points),
                            (next_section_id, roi_id),
                        )
                else:
                    annotation_points = getattr(self.wafer, annotation_type.name)[
                        section_id
                    ].points
                    propagated_points = GeometryCalculator.propagate_points(
                        self.wafer.sections[section_id].points,
                        annotation_points,
                        self.wafer.sections[next_section_id].points,
                    )
                    self.wafer.add(
                        annotation_type,
                        GeometryCalculator.points_to_poly(propagated_points),
                        next_section_id,
                    )
        self.wafer.wafer_to_manager()
        select_roi_from_name(str(self.wafer.sections[next_section_id]))


def move_fov(d, wafer):
    """Moves field of view of the image"""
    im = wafer.image
    canvas = im.getCanvas()
    r = canvas.getSrcRect()
    # adjust increment depending on zoom level
    increment = intr(40 / float(canvas.getMagnification()))
    xPixelIncrement = increment * (
        -1 if d is Direction.LEFT else 1 if d is Direction.RIGHT else 0
    )
    yPixelIncrement = increment * (
        -1 if d is Direction.UP else 1 if d is Direction.DOWN else 0
    )
    new_rectangle = Rectangle(
        min(max(0, r.x + xPixelIncrement), im.getWidth() - r.width),
        min(max(0, r.y + yPixelIncrement), im.getHeight() - r.height),
        r.width,
        r.height,
    )
    canvas.setSourceRect(new_rectangle)
    im.updateAndDraw()


def get_path(text):
    path = IJ.getFilePath(text)
    dlog("File selected: {}".format(path))
    return path


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
        return
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
            splitIndexes[1] = splitIndexes[1] + 1
            return range(*splitIndexes)
    elif userString.isdigit():
        return [int(userString)]
    return


def show_dialog(title, subtitle, suggestions, id_default):
    gd = GenericDialog(title)
    gd.addRadioButtonGroup(
        subtitle, suggestions, len(suggestions), 1, suggestions[id_default]
    )
    focus_on_ok(gd)
    gd.showDialog()
    if gd.wasCanceled():
        return
    return gd.getRadioButtonGroups()[0].getSelectedCheckbox()


def annotation_name_validation_dialog(name_suggestions):
    checkbox = show_dialog(
        "Validate the name of the annotation",
        "Name of the annotation",
        name_suggestions,
        0,
    )
    if not checkbox:
        dlog("No selection made: the section was not added")
        return
    if checkbox is None:
        return
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
def init_manager():
    """gets, places, resets the ROIManager"""
    manager = get_roi_manager()
    manager.setTitle("Annotations")
    manager.reset()
    manager.setSize(250, intr(0.95 * IJ.getScreenSize().height))
    manager.setLocation(IJ.getScreenSize().width - manager.getSize().width, 0)
    return manager


def get_roi_manager():
    manager = RoiManager.getInstance()
    if manager is None:
        manager = RoiManager()
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
    get_roi_manager().runCommand("Delete")


def delete_roi_by_index(index):
    manager = get_roi_manager()
    manager.select(index)
    manager.runCommand("Delete")


def get_roi_from_name(roi_name):
    manager = get_roi_manager()
    for roi in manager.getRoisAsArray():
        if roi.getName() == roi_name:
            return roi


def get_roi_index_from_name(name):
    manager = get_roi_manager()
    for id_roi, roi in enumerate(manager.getRoisAsArray()):
        if roi.getName() == name:
            return id_roi


def select_roi_from_name(roi_name):
    manager = get_roi_manager()
    for id_roi, roi in enumerate(manager.getRoisAsArray()):
        if roi.getName() == roi_name:
            manager.select(id_roi)


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
        return
    return roi_index


def toggle_fill(annotation_type):
    manager = get_roi_manager()
    for roi in manager.iterator():
        if annotation_type.string in roi.getName():
            if roi.getFillColor():
                roi.setFillColor(None)
            else:
                roi.setFillColor(annotation_type.color)
    manager.runCommand("Show All without labels")


def toggle_labels():
    manager = get_roi_manager()
    if manager.getDrawLabels():
        manager.runCommand("Show All without labels")
    else:
        manager.runCommand("UseNames", "true")
        manager.runCommand("Show All with labels")


# ----- End RoiManager functions ----- #


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


# convenience functions for HTML help message
def tag(text, tag):
    return "<{}>{}</{}>".format(tag, text, tag)


def print_list(*elements):
    a = "<ul>"
    for s in elements:
        a += "<li>{}</li>".format(s)
    a += "</ul>"
    return a


if __name__ == "__main__":
    L_HELP = (
        "[l] exports section transforms. Each transform transforms its section from global to local coordinates. "
        "If you apply the transforms to the global sections then:"
        + print_list(
            "the centroids of all transformed local sections are at (0,0)",
            "the transformed local sections are aligned",
        )
    )

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
            + print_list("[Shift] + mouse wheel", "[&larr] / [&rarr]"),
            "Toggle"
            + print_list(
                "[1] filling of sections",
                "[2] filling of rois",
                "[3] filling of focus",
                "[0] display labels",
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
            "[k] transfers annotations from a source wafer into this currently open target wafer (using landmarks for the transform)",
            (
                "[o]   (letter o) computes the section order that minimizes the travel of the microscope stage"
                " (not the same as the serial sectioning order of the sections)."
                ' Saves the order in the .magc file in the field "stage_order"'
            ),
            (
                "[j] computes the serial order based on the roi defined in the first section."
                " Updates the section positions."
                'Stores the serial order in the .magc file in the field "serial_order".'
                " We recommend to save beforehand a copy of the .magc file outside of the directory"
            ),
            L_HELP,
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
                ' Saves the order in the .magc file in the field "stage_order"'
            ),
            "[k] exports high resolution stack if companion high resolution image present",
            L_HELP,
        )
        + "<br><br><br></html>"
    )

    initial_ij_key_listeners = IJ.getInstance().getKeyListeners()
    wafer = Wafer()
    wafer.start_global_mode()
