from __future__ import with_statement

import itertools
import os
import sys
import threading
import time

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

SIZE_HANDLE = 15
LOCAL_SIZE_STANDARD = 400  # for local summary
MSG_DRAWN_ROI_MISSING = (
    "Please draw something before pressing [a]."
    + "\nAfter closing this message you can press [h] for help."
)


class Mode(object):
    GLOBAL = "global"
    LOCAL = "local"


class AnnotationTypeDef(object):
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


class Wafer(object):
    def __init__(self):
        self.mode = Mode.GLOBAL
        self.root = None
        self.manager = init_manager()
        self.image_path = self.init_image_path()
        self.magc_path = self.init_magc_path()
        self.init_images_global()

        self.sections = {}
        self.rois = {}
        self.focus = {}
        self.magnets = {}
        self.landmarks = {}
        self.transforms = {}
        self.poly_transforms = {}
        self.poly_transforms_inverse = {}
        self.serialorder = []
        self.tsporder = []
        self.tsp_orderer = TSPSolver(self.root)
        self.file_to_wafer()
        # self.name # TODO
        IJ.setTool("polygon")

    def init_image_path(self):
        try:
            self.root = os.path.normpath(
                DirectoryChooser("Select the experiment folder.").getDirectory()
            )
        except Exception:
            IJ.showMessage("Exit", "There was a problem accessing the folder")
            sys.exit("No directory was selected. Exiting.")
        if not os.path.isdir(self.root):
            IJ.showMessage("Exit", "No directory was selected. Exiting.")
            sys.exit("No directory was selected. Exiting.")
        wafer_im_names = sorted(
            [
                name
                for name in os.listdir(self.root)
                if any(
                    [
                        name.endswith(".tif"),
                        name.endswith(".png"),
                        name.endswith(".jpg"),
                        name.endswith(".tiff"),
                        name.endswith(".jpeg"),
                    ]
                )
                and not "verview" in name
            ]
        )
        if not wafer_im_names:
            IJ.showMessage(
                "Message",
                (
                    "There is no image (.tif, .png, .jpg, .jpeg, .tiff) in the experiment folder you selected."
                    + "\nAdd an image and start again the plugin."
                ),
            )
            sys.exit()
        else:
            return os.path.join(self.root, wafer_im_names[0],)

    def init_magc_path(self):
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
                cancel_msg="The substrate needs a name. Exiting.",
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
        config = ConfigParser.ConfigParser()
        with open(self.magc_path, "rb") as configfile:
            config.readfp(configfile)

        for header in config.sections():
            if "." in header:
                annotation_type, id = type_id(header, delimiter=".")
                for key, val in config.items(header):
                    if key in ["polygon", "location"]:
                        vals = [float(x) for x in val.split(",")]
                        points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                        self.add(
                            annotation_type, points_to_poly(points), id,
                        )

            elif header in ["serialorder", "tsporder"]:
                if config.get(header, header) != "[]":
                    setattr(
                        self,
                        header,
                        [int(x) for x in config.get(header, header).split(",")],
                    )
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

    def wafer_to_manager(self):
        """Draws all rois from the wafer into the manager"""
        self.manager.reset()
        if self.mode is Mode.GLOBAL:
            for landmark in self.landmarks.values():
                self.manager.addRoi(landmark.poly)
                landmark.poly.setHandleSize(AnnotationType.LANDMARK.handle_size_global)
            for section_id in sorted(self.sections.keys()):
                self.manager.addRoi(self.sections[section_id].poly)
                self.sections[section_id].poly.setHandleSize(
                    AnnotationType.SECTION.handle_size_global
                )
                for annotation_type in [
                    AnnotationType.ROI,
                    AnnotationType.FOCUS,
                    AnnotationType.MAGNET,
                ]:
                    annotations = getattr(self, annotation_type.name)
                    if section_id in annotations:
                        self.manager.addRoi(annotations[section_id].poly)
                        annotations[section_id].poly.setHandleSize(
                            annotation_type.handle_size_global
                        )
        elif self.mode is Mode.LOCAL:
            for id, section_id in enumerate(sorted(self.sections.keys())):
                for annotation_type in [
                    AnnotationType.SECTION,
                    AnnotationType.ROI,
                    AnnotationType.FOCUS,
                    AnnotationType.MAGNET,
                ]:
                    annotation = getattr(self, annotation_type.name).get(section_id)
                    if annotation is not None:
                        local_poly = transform_points_to_poly(
                            annotation.points, self.poly_transforms[section_id]
                        )
                        local_poly.setName(str(annotation))
                        local_poly.setStrokeColor(annotation_type.color)
                        local_poly.setImage(self.image)
                        local_poly.setPosition(0, id + 1, 0)
                        local_poly.setHandleSize(annotation_type.handle_size_local)
                        self.manager.addRoi(local_poly)

    def empty_annotations(self):
        self.sections = {}
        self.rois = {}
        self.focus = {}
        self.magnets = {}

    def empty_transforms(self):
        self.transforms = {}
        self.poly_transforms = {}
        self.poly_transforms_inverse = {}

    def manager_to_wafer(self):
        """Populates the wafer from the manager"""
        self.empty_annotations()
        for roi in self.manager.iterator():
            annotation_type, annotation_id = type_id(roi.getName())
            self.add(annotation_type, roi, annotation_id)
        self.empty_transforms()
        self.compute_transforms()

    def save(self):
        IJ.log("Saving ...")
        self.manager_to_wafer()
        config = ConfigParser.ConfigParser()

        for annotation_type in [
            AnnotationType.SECTION,
            AnnotationType.ROI,
            AnnotationType.FOCUS,
            AnnotationType.MAGNET,
            AnnotationType.LANDMARK,
        ]:
            annotations = getattr(self, annotation_type.name)
            if annotations:
                config.add_section(annotation_type.name)
                config.set(
                    annotation_type.name, "number", str(len(annotations)),
                )
                for id in sorted(annotations.keys()):
                    header = "{}.{:04}".format(annotation_type.name, id)
                    config.add_section(header)

                    if annotation_type in [
                        AnnotationType.SECTION,
                        AnnotationType.ROI,
                        AnnotationType.FOCUS,
                    ]:
                        config.set(
                            header,
                            "polygon",
                            points_to_flat_string(annotations[id].points),
                        )
                        if annotation_type in [
                            AnnotationType.SECTION,
                            AnnotationType.ROI,
                        ]:
                            config.set(
                                header,
                                "center",
                                point_to_flat_string(annotations[id].centroid),
                            )
                            config.set(header, "area", str(annotations[id].area))
                            config.set(
                                header,
                                "angle",
                                str(((annotations[id].area - 90) % 360) - 180),
                            )
                    elif annotation_type in [
                        AnnotationType.MAGNET,
                        AnnotationType.LANDMARK,
                    ]:
                        config.set(
                            header,
                            "location",
                            point_to_flat_string(annotations[id].centroid),
                        )
                config.add_section("end_{}".format(annotation_type.name))

        for order_name in ["serialorder", "tsporder"]:
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

    def save_csv(self):
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
                                self.rois[section_id].centroid[0]
                                if section_id in self.rois.keys()
                                else "",
                                self.rois[section_id].centroid[1]
                                if section_id in self.rois.keys()
                                else "",
                                self.rois[section_id].angle
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
                                self.tsporder[id] if len(self.tsporder) > id else "",
                                self.serialorder[id]
                                if len(self.serialorder) > id
                                else "",
                            ]
                        ],
                    )
                )
                # handle case: more landmarks than sections
                for i in range(len(self.sections), len(self.landmarks)):
                    f.write(
                        ",,,,,,,,,{},{},,".format(
                            self.landmarks[i][0], self.landmarks[i][1]
                        )
                    )

                f.write("\n")
        IJ.log("Saved to {}".format(csv_path))

    def close_mode(self):
        self.save()
        if self.mode is Mode.GLOBAL:
            self.image.hide()
        else:
            self.image.close()
        self.manager.reset()
        # restore the keylisteners of the IJ window
        map(IJ.getInstance().addKeyListener, initial_ij_key_listeners)

    def start_local_mode(self):
        IJ.log("Starting local mode ...")
        self.mode = Mode.LOCAL
        self.compute_transforms()  # 0.006s
        self.create_local_stack()  # 0.5s
        self.wafer_to_manager()  # 0.1s
        self.set_listeners()
        self.manager.runCommand("UseNames", "false")
        self.manager.runCommand("Show None")  # 0.0s
        set_roi_and_update_roi_manager(0)  # select first ROI
        self.arrange_windows()  # 0.3

    def compute_transforms(self):
        _, _, display_size, _ = self.get_display_parameters()
        self.local_display_size = display_size

        for id in self.sections:
            # image transform
            aff = AffineTransform2D()
            aff.translate([-v for v in self.sections[id].centroid])
            aff.rotate(self.sections[id].angle * Math.PI / 180)
            self.transforms[id] = aff

            # poly transform (there is an offset)
            aff_copy = aff.copy()
            poly_translation = AffineTransform2D()
            poly_translation.translate([0.5 * v for v in self.local_display_size])
            self.poly_transforms[id] = aff_copy.preConcatenate(poly_translation)
            self.poly_transforms_inverse[id] = self.poly_transforms[id].inverse()

    def create_local_stack(self):
        imgs = [
            Views.interval(
                RV.transform(
                    Views.interpolate(
                        Views.extendZero(self.img_global), NLinearInterpolatorFactory()
                    ),
                    self.transforms[id],
                ),
                [-int(0.5 * v) for v in self.local_display_size],
                [int(0.5 * v) for v in self.local_display_size],
            )
            for id in sorted(self.sections)
        ]
        self.img_local = Views.permute(
            Views.addDimension(Views.stack(imgs), 0, 0), 3, 2
        )
        IL.show(self.img)
        self.image_local = IJ.getImage()

    def set_listeners(self):
        add_key_listener_everywhere(KeyListener())
        add_mouse_wheel_listener_everywhere(MouseWheelListener())

    def add(self, annotation_type, poly, annotation_id, template=None):
        if self.mode is Mode.GLOBAL:
            getattr(self, annotation_type.name)[annotation_id] = Annotation(
                annotation_type, poly, annotation_id
            )
        else:
            getattr(self, annotation_type.name)[annotation_id] = Annotation(
                annotation_type,
                transform_points_to_poly(
                    poly_to_points(poly), self.poly_transforms_inverse[annotation_id]
                ),
                annotation_id,
            )

    def init_images_global(self):
        self.image_global = IJ.openImage(self.image_path)
        self.img_global = IL.wrap(self.image)

    @property
    def image(self):
        if self.mode is Mode.GLOBAL:
            return self.image_global
        else:
            return self.image_local

    @property
    def img(self):
        if self.mode is Mode.GLOBAL:
            return self.image_global
        else:
            return self.img_local

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

    def suggest_ids(self, annotation_type):
        """[1,3,4] -> [0,2,5]"""
        item_ids = [item.id for item in getattr(self, annotation_type.name).values()]
        if not item_ids:
            return [0]
        return sorted(set(range(max(item_ids) + 2)) - set(item_ids))

    def get_closest(self, annotation_type, point):
        distances = sorted(
            [
                [get_distance(point, item.centroid), item]
                for item in getattr(self, annotation_type.name).values()
            ]
        )
        return [d[1] for d in distances]

    def get_display_parameters(self):
        """calculate [display_size, crop_size, tissue_magnet_distance] based on sectionSize"""
        tissue_magnet_distance = 0

        section_extent = 0
        for section in self.sections.values()[: min(5, len(self.sections))]:
            section_extent = max(section_extent, longest_diagonal(section.points))

        display_size = [
            int(1.2 * section_extent),
            int(1.2 * section_extent)
            if not self.magnets
            else int(1.4 * section_extent),
        ]
        display_center = [int(0.5 * display_size[0]), int(0.5 * display_size[1])]
        crop_size = [2 * display_size[0], 2 * display_size[1]]

        if self.magnets:
            tissue_magnet_distances = []
            for magnet_id in sorted(self.magnets.keys()[: min(5, len(self.magnets))]):
                tissue_magnet_distances.append(
                    get_distance(
                        self.sections[magnet_id].centroid,
                        self.magnets[magnet_id].centroid,
                    )
                )
            tissue_magnet_distance = sum(tissue_magnet_distances) / len(
                tissue_magnet_distances
            )
        return display_center, tissue_magnet_distance, display_size, crop_size

    def compute_stage_tsp_order(self):
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
        self.tsporder = self.tsp_orderer.compute_tsp_order(distances)


class Annotation(object):
    def __init__(self, annotation_type, poly, id, template=None):
        self._type = annotation_type
        self.poly = poly
        self.id = id
        self.points = poly_to_points(poly)
        self.centroid = self.compute_centroid()
        self.area = poly.getStatistics().pixelCount
        self.angle = self.compute_angle()
        self.template = template
        self.set_poly_properties()

    def __str__(self):
        return "{}-{:04}".format(self._type.string, self.id)

    def __len__(self):
        return self.poly.size()

    def set_poly_properties(self):
        self.poly.setName(str(self))
        self.poly.setStrokeColor(self._type.color)

    def contains(self, point):
        return self.poly.containsPoint(*point)

    def compute_centroid(self):
        if len(self) == 1:
            return self.points[0]
        return list(points_to_poly(self.points).getContourCentroid())

    def compute_angle(self):
        if len(self) < 2:
            return None
        return self.poly.getFloatAngle(
            self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1],
        )


# ----- Listeners and handlers ----- #
def add_key_listener_everywhere(my_listener):
    for elem in (
        [IJ.getImage().getWindow(), IJ.getImage().getWindow().getCanvas(),]
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
        [IJ.getImage().getWindow(), IJ.getImage().getWindow().getCanvas(),]
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
        event.consume()
        if wafer.mode is Mode.GLOBAL:
            handle_key_global(event)
        elif wafer.mode is Mode.LOCAL:
            handle_key_local(event)


def handle_key_global(keyEvent):
    keycode = keyEvent.getKeyCode()
    if keycode == KeyEvent.VK_A:
        handle_key_a()
    if keycode == KeyEvent.VK_S:
        wafer.save()
    if keycode == KeyEvent.VK_T:
        if wafer.sections:
            wafer.close_mode()
            wafer.start_local_mode()
        else:
            IJ.showMessage(
                "Cannot toggle to local mode because there are no sections defined."
            )
    if keycode == KeyEvent.VK_Q:  # terminate and save
        wafer.manager_to_wafer()  # will be repeated in close_mode but it's OK
        wafer.compute_stage_tsp_order()
        wafer.close_mode()
    if keycode == KeyEvent.VK_O:
        wafer.manager_to_wafer()
        wafer.compute_stage_tsp_order()
        wafer.save()
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


def handle_key_m_global():
    manager = wafer.manager
    for roi in manager.iterator():
        annotation_type, annotation_id = type_id(roi.getName())
        if annotation_type is AnnotationType.SECTION:
            roi.setName(str(annotation_id))
            roi.setStrokeWidth(8)
        else:
            roi.setName("")
            roi.setStrokeWidth(1)
    IJ.run(
        "Labels...",
        (
            "color=white font="
            + str(int(wafer.image.getWidth() / 400))
            + " show use draw bold"
        ),
    )
    flattened = wafer.image.flatten()
    flattened_path = os.path.join(wafer.root, "overview_global.jpg")
    IJ.save(flattened, flattened_path)
    IJ.log("Flattened global image saved to {}".format(flattened_path))
    flattened.close()
    wafer.start_global_mode()


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
    if keycode == KeyEvent.VK_Q:
        wafer.manager_to_wafer()  # will be repeated in close_mode but it's OK
        wafer.compute_stage_tsp_order()
        wafer.close_mode()
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
    if keycode == KeyEvent.VK_S:
        wafer.save()
    if keycode == KeyEvent.VK_O:
        wafer.manager_to_wafer()
        wafer.compute_stage_tsp_order()
        wafer.save()
    if keycode == KeyEvent.VK_H:
        IJ.showMessage("Help for local mode", HELP_MSG_LOCAL)
    keyEvent.consume()


def handle_key_m_local():
    montageMaker = MontageMaker()
    stack = wafer.image.getStack()
    n_slices = wafer.image.getNSlices()

    n_rows = int(n_slices ** 0.5)
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
        handle_size = 5 * int(im_w / LOCAL_SIZE_STANDARD)
        stroke_size = 3 * im_w / LOCAL_SIZE_STANDARD

    flattened_ims = []
    for id, section_id in enumerate(sorted(wafer.sections.keys())):
        im_p = stack.getProcessor(id + 1).duplicate()
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
    manager = wafer.manager
    selected_indexes = manager.getSelectedIndexes()
    if len(selected_indexes) != 1:
        IJ.showMessage(
            "Warning",
            "To delete an annotation with [x] an annotation must be selected in blue in the annotation manager",
        )
        return

    selected_poly = manager.getRoi(selected_indexes[0])
    poly_name = selected_poly.getName()
    annotation_type, annotation_id = type_id(poly_name)

    if annotation_type in [
        AnnotationType.MAGNET,
        AnnotationType.ROI,
        AnnotationType.FOCUS,
    ]:
        if get_OK("Delete {}?".format(poly_name)):
            delete_selected_roi()
            wafer.image.killRoi()
            del getattr(wafer, annotation_type.name)[annotation_id]
            manager.select(
                get_roi_index_by_name(str(wafer.sections[annotation_id]))
            )  # select the section
    elif annotation_type is AnnotationType.SECTION:
        section_roi_index = get_roi_index_by_name(str(wafer.sections[annotation_id]))
        linked_annotation_types = []
        message = ""
        for _type in [
            AnnotationType.ROI,
            AnnotationType.FOCUS,
            AnnotationType.MAGNET,
        ]:
            linked_annotation = getattr(wafer, _type.name).get(annotation_id)
            if linked_annotation:
                message += "{}\n \n".format(str(linked_annotation))
                linked_annotation_types.append(_type)
        message = "".join(
            [
                "Delete {}?".format(poly_name),
                "\n \nIt will also delete\n \n" if message else "",
                message,
            ]
        )
        if get_OK(message):
            if wafer.image.getNSlices() == 1:
                if get_OK(
                    "Case not yet handled: you are trying to delete the only existing section."
                    + "\n\nFiji will close. Please delete the .ini file and start over from scratch instead.\n\nContinue?"
                ):
                    wafer.image.close()
                    manager.close()
                    sys.exit()
                else:
                    return
            wafer.image.killRoi()
            for linked_annotation_type in linked_annotation_types:
                index = get_roi_index_by_name(
                    str(getattr(wafer, linked_annotation_type.name)[annotation_id])
                )
                delete_roi_by_index(index)
                del getattr(wafer, linked_annotation_type.name)[annotation_id]
            del wafer.sections[annotation_id]
            del wafer.transforms[annotation_id]
            del wafer.poly_transforms[annotation_id]
            del wafer.poly_transforms_inverse[annotation_id]

            wafer.image.close()
            manager.reset()
            wafer.start_local_mode()

            # select the next section
            if section_roi_index < manager.getCount():
                set_roi_and_update_roi_manager(section_roi_index)
            else:
                set_roi_and_update_roi_manager(manager.getCount() - 1)


def handle_key_p_local():
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
    annotation_type, annotation_id = type_id(poly_name)
    if annotation_type is AnnotationType.SECTION:
        IJ.showMessage(
            "Info",
            "Sections cannot be propagated. Only rois, focus, magnets can be propagated",
        )
        return

    min_section_id = min(wafer.sections.keys())
    max_section_id = max(wafer.sections.keys())

    gd = GenericDialogPlus("Propagation")
    gd.addMessage(
        "This {} is defined in section number {}.\nTo what sections do you want to propagate this {}?".format(
            annotation_type.string, annotation_id, annotation_type.string
        )
    )
    gd.addStringField(
        "Enter a range or single values separated by commas. "
        + "Range can be start-end (4-7 = 4,5,6,7) or "
        + "start-end-increment (2-11-3 = 2,5,8,11).",
        "{}-{}".format(min_section_id, max_section_id),
    )
    gd.addButton(
        "All sections {}-{}".format(min_section_id, max_section_id), ButtonClick(),
    )
    gd.addButton(
        "First half of the sections {}-{}".format(
            min_section_id, int(max_section_id / 2.0)
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
    valid_input_indexes = [i for i in input_indexes if i in wafer.sections.keys()]
    if not valid_input_indexes:
        return

    # changing the mode temporarily for the wafer.add
    # (adding a poly in global coordinates)
    wafer.mode = Mode.GLOBAL
    for input_index in valid_input_indexes:
        propagated_points = propagate_points(
            wafer.sections[annotation_id].points,
            getattr(wafer, annotation_type.name)[annotation_id].points,
            wafer.sections[input_index].points,
        )
        wafer.add(
            annotation_type,
            points_to_poly(propagated_points),
            input_index,
            template=annotation_id,
        )
    wafer.mode = Mode.LOCAL

    wafer.wafer_to_manager()
    # select the first propagated roi
    select_roi_by_name(
        "{}-{:04}".format(annotation_type.string, valid_input_indexes[0])
    )


def handle_key_a():
    drawn_roi = wafer.image.getRoi()
    if not drawn_roi:
        IJ.showMessage(
            "Info", MSG_DRAWN_ROI_MISSING,
        )
        return
    if drawn_roi.getState() == PolygonRoi.CONSTRUCTING and drawn_roi.size() > 3:
        return

    name_suggestions = []
    if wafer.mode is Mode.LOCAL:
        slice_id = wafer.image.getSlice()
        section_id = sorted(wafer.sections.keys())[slice_id - 1]

        if drawn_roi.size() == 2:
            name_suggestions.append("magnet-{:04}".format(section_id))
        else:
            name_suggestions.append("section-{:04}".format(section_id))
            name_suggestions.append("roi-{:04}".format(section_id))
        name_suggestions.append("focus-{:04}".format(section_id))
    elif wafer.mode is Mode.GLOBAL:
        closest_sections = wafer.get_closest(
            AnnotationType.SECTION, drawn_roi.getContourCentroid()
        )
        if drawn_roi.size() == 2:
            name_suggestions += [
                "magnet-{:04}".format(section.id) for section in closest_sections[:3]
            ]
            name_suggestions += [
                "landmark-{:04}".format(id)
                for id in wafer.suggest_ids(AnnotationType.LANDMARK)
            ]
        else:
            name_suggestions += [
                "section-{:04}".format(id)
                for id in wafer.suggest_ids(AnnotationType.SECTION)
            ]
            name_suggestions += [
                "roi-{:04}".format(section.id) for section in closest_sections[:3]
            ]

    annotation_name = annotation_name_validation_dialog(name_suggestions)
    if annotation_name is None:
        return

    points = poly_to_points(drawn_roi)
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
    annotation_type, annotation_id = type_id(annotation_name)
    if wafer.mode is Mode.LOCAL:
        drawn_roi.setHandleSize(annotation_type.handle_size_local)
    else:
        drawn_roi.setHandleSize(annotation_type.handle_size_global)
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
    im = wafer.image
    canvas = im.getCanvas()
    r = canvas.getSrcRect()
    # adjust increment depending on zoom level
    increment = int(40 / float(canvas.getMagnification()))
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


# --- End Key listeners --- #
# ----- End Listeners and handlers ----- #
# ----- multithreading ----- #
def start_threads(function, fractionCores=1, arguments=None, nThreads=None):
    threads = []
    if nThreads == None:
        threadRange = range(
            max(int(Runtime.getRuntime().availableProcessors() * fractionCores), 1)
        )
    else:
        threadRange = range(nThreads)
    IJ.log("Running in parallel with ThreadRange = {}".format(threadRange))
    for p in threadRange:
        if arguments == None:
            thread = threading.Thread(target=function)
        else:
            thread = threading.Thread(group=None, target=function, args=arguments)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


# ----- End multithreading ----- #


# ----- Dialogs ----- #
def get_name(text, default_name="", cancel_msg=None):
    gd = GenericDialog(text)
    gd.addStringField(text, default_name)
    gd.showDialog()
    if gd.wasCanceled() and cancel_msg is not None:
        IJ.showMessage(cancel_msg)
        sys.exit()
    return gd.getNextString()


def get_OK(text):
    gd = GenericDialog("User prompt")
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
        subtitle, suggestions, len(suggestions), 1, suggestions[id_default],
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


# ----- End Dialogs ----- #

# ----- RoiManager functions ----- #
def get_roi_manager():
    manager = RoiManager.getInstance()
    if manager == None:
        manager = RoiManager()
    return manager


def init_manager():
    # get, place, and reset the ROIManager
    manager = get_roi_manager()
    manager.setTitle("Annotations")
    manager.reset()
    manager.setSize(
        250, int(0.95 * IJ.getScreenSize().height)
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
    scrollBar.setValue(int(1.5 * barMax))


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


def select_roi_by_name(roiName):
    manager = get_roi_manager()
    roiIndex = [roi.getName() for roi in manager.getRoisAsArray()].index(roiName)
    manager.select(roiIndex)


def set_roi_and_update_roi_manager(roi_index, select=True):
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


def type_id(name, delimiter="-"):
    """roi-0012 -> AnnotationType.ROI, 12"""
    id = int(name.split(delimiter)[-1])
    annotation_type = [
        v
        for v in AnnotationType.__dict__.values()
        if (type(v) is AnnotationTypeDef) and (v.string in name)
    ][0]
    return annotation_type, id


# ----- End RoiManager functions ----- #
# ----- HTML functions ----- #
def tag(text, tag):
    return "<{}>{}</{}>".format(tag, text, tag)


def print_list(*elements):
    a = "<ul>"
    for s in elements:
        a += "<li>{}</li>".format(s)
    a += "</ul>"
    return a


# ----- End HTML functions ----- #
# ----- Geometric functions ----- #
def get_distance(p_1, p_2):
    return Math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)


def longest_diagonal(points):
    max_diag = 0
    for p1 in points:
        for p2 in points:
            max_diag = max(get_distance(p1, p2), max_diag)
    return int(max_diag)


def points_to_poly(points):
    if len(points) == 1:
        return PointRoi(*points[0])
    if len(points) == 2:
        polygon_type = PolygonRoi.POLYLINE
    elif len(points) > 2:
        polygon_type = PolygonRoi.POLYGON
    return PolygonRoi(
        [point[0] for point in points], [point[1] for point in points], polygon_type,
    )


def poly_to_points(poly):
    float_polygon = poly.getFloatPolygon()
    return [[x, y] for x, y in zip(float_polygon.xpoints, float_polygon.ypoints)]


def points_to_flat_string(points):
    points_flat = []
    for point in points:
        points_flat.append(point[0])
        points_flat.append(point[1])
    points_string = ",".join([str(round(x, 3)) for x in points_flat])
    return points_string


def point_to_flat_string(point):
    flat_string = ",".join([str(round(x, 3)) for x in point])
    return flat_string


def points_to_flat(points):
    flat = []
    for point in points:
        flat += point
    return flat


def flat_to_poly(flat):
    if len(flat) == 2:
        poly = Point(flat[::2], flat[1::2])
    if len(flat) == 4:
        poly = PolygonRoi(flat[::2], flat[1::2], PolygonRoi.POLYLINE)
    else:
        poly = PolygonRoi(flat[::2], flat[1::2], PolygonRoi.POLYGON)
    return poly


def transform_points_to_poly(source_points, aff):
    target_points = []
    for source_point in source_points:
        target_point = jarray.array([0, 0], "d")
        aff.apply(source_point, target_point)
        target_points.append(target_point)
    return points_to_poly(target_points)


def affine_t(x_in, y_in, x_out, y_out):
    X = Matrix(
        jarray.array(
            [[x, y, 1] for (x, y) in zip(x_in, y_in)], java.lang.Class.forName("[D")
        )
    )
    Y = Matrix(
        jarray.array(
            [[x, y, 1] for (x, y) in zip(x_out, y_out)], java.lang.Class.forName("[D")
        )
    )
    aff = X.solve(Y)
    return aff


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


def invert_affine_t(aff):
    return aff.inverse()


def rigid_t(x_in, y_in, x_out, y_out):
    rigidModel = RigidModel2D()
    pointMatches = HashSet()
    for x_i, y_i, x_o, y_o in zip(x_in, y_in, x_out, y_out):
        pointMatches.add(PointMatch(Point([x_i, y_i]), Point([x_o, y_o]),))
    rigidModel.fit(pointMatches)
    return rigidModel


def apply_rigid_t(x_in, y_in, rigid_model):
    x_out = []
    y_out = []
    for x_i, y_i in zip(x_in, y_in):
        x_o, y_o = rigid_model.apply([x_i, y_i])
        x_out.append(x_o)
        y_out.append(y_o)
    return x_out, y_out


def points_to_xy(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return x, y


def propagate_points(source_section, source_points, target_section):
    source_section_x, source_section_y = points_to_xy(source_section)
    source_points_x, source_points_y = points_to_xy(source_points)
    target_section_x, target_section_y = points_to_xy(target_section)
    if len(source_section) == 2:
        compute_t = rigid_t
        apply_t = apply_rigid_t
    else:
        compute_t = affine_t
        apply_t = apply_affine_t

    trans = compute_t(
        source_section_x, source_section_y, target_section_x, target_section_y
    )
    target_points_x, target_points_y = apply_t(source_points_x, source_points_y, trans)
    target_points = [[x, y] for x, y in zip(target_points_x, target_points_y)]
    return target_points


# ----- End Geometric functions ----- #


def pairwise(iterable):
    """from https://docs.python.org/3/library/itertools.html#itertools.pairwise
    'ABCD' --> AB BC CD"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


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
            "Downloading Windows-10 precompiled cygwin1.dll from \n \n{}\n \n"
            "That file is needed to compute the section order that minimizes stage travel."
            "\n \nDo you agree? "
        )
        if "windows" in System.getProperty("os.name").lower():
            # download cygwin1.dll
            if not os.path.isfile(cygwindll_path):
                # download cygwin1.dll
                if get_OK(download_msg.format(cygwindll_url)):
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
                    if get_OK(download_msg.format(url)):
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
                order = [
                    int(x)
                    for x in f.readlines()[1].replace("\n", "").split(" ").remove("")
                ]
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
                "The total cost of the {}optimized order is {}".format(name, int(cost))
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
                distances.append(int(float(distance)))
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
            IJ.log(
                "The optimal section order to minimize stage travel is: {}".format(
                    order
                )
            )
        except (Exception, java_exception) as e:
            IJ.log(
                "The path to minimize stage movement could not be computed: {}".format(
                    e
                )
            )
            return []
        return order


if __name__ == "__main__":
    HELP_MSG_GLOBAL = (
        "<html><br>[a] = Press the key a<br><br>"
        + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
        + tag("Navigation", "h3")
        + print_list(
            "Zoom in/out" + print_list("[Ctrl] + Mouse wheel", "[+] / [-]",),
            "Move up/down" + print_list("Mouse wheel", "&uarr / &darr"),
            "Move left/right" + print_list("[Shift] + Mouse wheel", "&larr / &rarr"),
            "Toggle filling of"
            + print_list("sections   [1]", "rois   [2]", "focus   [3]"),
            "[0] (number 0) toggle labels",
        )
        + tag("Actions", "h3")
        + print_list(
            "Mouse click"
            + print_list(
                "Left - draw point",
                "Right - join first and last points to close the polygon",
            ),
            "[escape] Stop current drawing",
            "[a] add an annotation that you have drawn",
            "[t] toggle to local mode",
            "[q] quit (everything will be saved)",
            "[s] save (happens automatically when toggling [t] and quitting [q])",
            "[m] export a summary image of global mode",
            (
                "[o]   (letter o) compute the section order that minimizes the travel of the microscope stage"
                " (not the same as the serial sectioning order of the sections)."
                ' The order is saved in the field "tsporder" (stands for Traveling Salesman Problem order)'
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
        + tag("Actions", "h3")
        + print_list(
            "[a] create/modify an annotation",
            "[t] toggle to global mode",
            "[g] validate your modification. It happens automatically when browsing through the sections with [d]/[f], [c]/[v], [e]/[r], or with the mouse wheel",
            "[q] quit. Everything will be saved",
            "[s] save to file (happens already automatically when toggling [t] or quitting [q])",
            "[m] export a summary montage",
            (
                "[o]   (letter o) compute the section order that minimizes the travel of the microscope stage"
                " (not the same as the serial sectioning order of the sections)."
                ' The order is saved in the field "tsporder" (stands for Traveling Salesman Problem order)'
            ),
        )
        + "<br><br><br></html>"
    )

    initial_ij_key_listeners = IJ.getInstance().getKeyListeners()
    wafer = Wafer()
    wafer.start_global_mode()
