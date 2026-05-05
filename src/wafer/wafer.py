import itertools
import os
import sys
import time
from contextlib import contextmanager

import ConfigParser
from ij import IJ, ImagePlus, ImageStack
from ij.gui import OvalRoi, PointRoi, Roi
from ij.io import DirectoryChooser
from ij.plugin import MontageMaker
from java.awt.geom.Point2D import Float as pFloat
from java.lang import Math
from net.imglib2.img.display.imagej import ImageJFunctions as IL
from net.imglib2.interpolation.randomaccess import NLinearInterpolatorFactory
from net.imglib2.realtransform import AffineTransform2D
from net.imglib2.realtransform import RealViews as RV
from net.imglib2.view import Views

from src.geometry.geometry import (
    affine_t,
    angle_to_string,
    apply_affine_t,
    change_basis,
    get_distance,
    longest_diagonal,
    point_to_flat_string,
    points_to_flat_string,
    points_to_poly,
    poly_to_points,
    propagate_points,
    transform_points_to_poly,
    transform_poly,
    xy_to_points,
)
from src.geometry.transforms import (
    TRANSFORM_DISPLAY,
    TRANSFORM_DISPLAY_INVERSE,
    TRANSFORM_SUBPIXEL,
    TRANSFORM_SUBPIXEL_INVERSE,
)
from src.order.magreorderer import (
    MagReorderer,
    REORDER_ROI_ID,
    compute_highres_roi_size,
    export_highres,
    get_downsampling_factor,
)
from src.order.neighbor import Neighbor
from src.order.tspsolver import compute_tsp_order, ensure_tsp_solvers
from src.ui.dialog import (
    ask_export_channel,
    get_name,
    get_OK,
    get_path,
    user_confirmation,
)
from src.ui.keylistener import KeyListener
from src.ui.mode import Mode
from src.ui.mousewheellistener import MouseWheelListener
from src.ui.ui import add_key_listener_everywhere, add_mouse_wheel_listener_everywhere
from src.utils.file import get_image_paths, mkdir_p
from src.utils.log import dlog
from src.utils.matrix import init_mat
from src.utils.numeric import get_square_row_col, intr, suggest_ids
from src.utils.roimanager import (
    delete_roi_by_index,
    delete_selected_roi,
    get_roi_index_from_name,
    init_manager,
    set_roi_and_update_roi_manager,
)
from src.wafer.focus import Focus
from src.wafer.landmark import Landmark
from src.wafer.magnet import Magnet
from src.wafer.parse import parse_annotation_name
from src.wafer.roi import ROI
from src.wafer.section import Section

DEBUG = False
DISPLAY_FACTOR = 1.2
LOCAL_SIZE_STANDARD = 400


class Wafer(object):
    def __init__(self, magc_path=None):
        self.initial_ij_key_listeners = IJ.getInstance().getKeyListeners()
        self.mode = Mode.GLOBAL
        self.root, self.image_path = self.init_image_path(magc_path)
        ensure_tsp_solvers()
        self.manager = init_manager()
        self.magc_path = magc_path if magc_path is not None else self.init_magc_path()
        self.init_images_global()
        self.sections = {}
        self.landmarks = {}
        self.local_display_size = None
        self.serial_order = []
        self.stage_order = []
        self.file_to_wafer()
        IJ.setTool("polygon")

    def __len__(self):
        """The number of sections."""
        return len(self.sections)

    @property
    def image(self):
        if self.mode is Mode.GLOBAL:
            return self.image_global
        return self.image_local

    @property
    def landmarks_xy(self):
        return [self.landmarks[key].centroid for key in sorted(self.landmarks)]

    @contextmanager
    def set_mode(self, mode):
        """Temporarily sets mode and reverts it."""
        old_mode = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = old_mode

    def set_listeners(self):
        add_key_listener_everywhere(KeyListener(self), self.mode)
        add_mouse_wheel_listener_everywhere(MouseWheelListener(self))

    def to_global(self, section_id, poly):
        """Converts poly from local to global coordinates (no-op in global mode)."""
        if self.mode is Mode.GLOBAL:
            return poly
        return transform_points_to_poly(
            poly_to_points(poly),
            self.sections[section_id].poly_transform_inverse,
        )

    # ------------------------------------------------------------------
    # add_* methods
    # ------------------------------------------------------------------

    def add_section(self, id_, poly):
        """Adds (or replaces) a section. poly must be in global coordinates."""
        if id_ not in self.serial_order:
            self.serial_order.append(id_)
            self.stage_order.append(id_)
        self.sections[id_] = Section(id_, poly)
        return self.sections[id_]

    def add_roi(self, id_section, id_roi, poly):
        """Adds a ROI to a section, converting from local coords if needed."""
        poly = self.to_global(id_section, poly)
        roi = ROI(id_section, id_roi, poly)
        self.sections[id_section].rois[id_roi] = roi
        return roi

    def add_focus(self, id_section, poly):
        """Adds a focus annotation to a section, converting from local coords if needed."""
        poly = self.to_global(id_section, poly)
        focus = Focus(id_section, poly)
        self.sections[id_section].focus = focus
        return focus

    def add_magnet(self, id_section, poly):
        """Adds a magnet annotation to a section, converting from local coords if needed."""
        poly = self.to_global(id_section, poly)
        magnet = Magnet(id_section, poly)
        self.sections[id_section].magnet = magnet
        return magnet

    def add_landmark(self, id_, poly):
        """Adds a landmark (always in global coordinates)."""
        landmark = Landmark(id_, poly)
        self.landmarks[id_] = landmark
        return landmark

    def propagate_annotation(self, cls, section_id, annotation_id, target_ids):
        """Propagates a single annotation to a list of target sections."""
        section = self.sections[section_id]
        if cls is ROI:
            annotation_points = section.rois[annotation_id].points
        elif cls is Focus:
            annotation_points = section.focus.points
        elif cls is Magnet:
            annotation_points = section.magnet.points
        with self.set_mode(Mode.GLOBAL):
            for target_id in target_ids:
                propagated_points = propagate_points(
                    section.points,
                    annotation_points,
                    self.sections[target_id].points,
                )
                if cls is ROI:
                    self.add_roi(
                        target_id, annotation_id, points_to_poly(propagated_points)
                    )
                elif cls is Focus:
                    self.add_focus(target_id, points_to_poly(propagated_points))
                elif cls is Magnet:
                    self.add_magnet(target_id, points_to_poly(propagated_points))
        self.wafer_to_manager()

    def propagate_section_to_neighbor(self, neighbor):
        """Propagates all annotations of the current section to the neighboring serial section.

        Returns the neighbor section name (for UI selection), or None if no neighbor exists.
        """
        self.manager.runCommand("Update")
        self.manager_to_wafer()
        slice_id = self.image.getSlice()
        section_id = self.serial_order[slice_id - 1]
        try:
            id_section_neighbor = self.serial_order[
                slice_id if neighbor is Neighbor.NEXT else slice_id - 2
            ]
        except IndexError:
            dlog(
                "Cannot propagate to the {} section: there is no {} section".format(
                    neighbor, neighbor
                )
            )
            return None
        section = self.sections[section_id]
        neighbor_section = self.sections[id_section_neighbor]

        def propagate(annotation_points):
            return points_to_poly(
                propagate_points(
                    section.points, annotation_points, neighbor_section.points
                )
            )

        with self.set_mode(Mode.GLOBAL):
            if section.focus is not None:
                self.add_focus(id_section_neighbor, propagate(section.focus.points))
            if section.magnet is not None:
                self.add_magnet(id_section_neighbor, propagate(section.magnet.points))
            for roi_id, roi in section.rois.iteritems():
                self.add_roi(id_section_neighbor, roi_id, propagate(roi.points))
        self.wafer_to_manager()
        return str(neighbor_section)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_image_path(self, magc_path):
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
        paths = get_image_paths(root)
        if not paths:
            if magc_path is None:
                IJ.showMessage(
                    "Exit",
                    "There is no image (.tif, .png, .jpg, .jpeg, .tiff) in the experiment folder you selected."
                    "\nAdd an image and start again the plugin",
                )
                sys.exit("No image in folder. Exiting.")
            return root, None
        return root, paths[0]

    def init_magc_path(self):
        """Loads existing .magc file or creates a new one if it does not exist."""
        for filename in os.listdir(self.root):
            if filename.endswith(".magc"):
                return os.path.join(self.root, filename)

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
        return magc_path

    def init_images_global(self):
        if self.image_path is None:
            dlog("Warning: images not properly initialized")
            return
        self.image_global = IJ.openImage(self.image_path)
        self.img_global = IL.wrap(self.image_global)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def file_to_wafer(self):
        """Populates the wafer instance from the .magc file."""
        config = ConfigParser.ConfigParser()
        with open(self.magc_path, "rb") as configfile:
            config.readfp(configfile)
        for header in config.sections():
            if "." in header:
                cls, section_id, annotation_id = parse_annotation_name(
                    header, delimiter_0="."
                )
                for key, val in config.items(header):
                    if key not in ("polygon", "location"):
                        continue
                    vals = [float(x) for x in val.split(",")]
                    points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                    poly = points_to_poly(points)
                    if cls is Section:
                        self.add_section(section_id, poly)
                    elif cls is ROI:
                        self.add_roi(section_id, annotation_id, poly)
                    elif cls is Focus:
                        self.add_focus(section_id, poly)
                    elif cls is Magnet:
                        self.add_magnet(section_id, poly)
                    elif cls is Landmark:
                        self.add_landmark(section_id, poly)
            elif (
                header in ("serial_order", "stage_order")
                and config.get(header, header) != "[]"
            ):
                order = [int(x) for x in config.get(header, header).split(",")]
                setattr(self, header, order)
        if not self.serial_order:
            self.serial_order = sorted(self.sections.keys())
        if not self.stage_order:
            self.stage_order = sorted(self.sections.keys())
        n_rois = sum(len(s.rois) for s in self.sections.values())
        n_focuses = sum(s.focus is not None for s in self.sections.values())
        n_magnets = sum(s.magnet is not None for s in self.sections.values())
        dlog(
            (
                "File successfully read with \n{} sections \n{} rois \n{} focus"
                + "\n{} magnets \n{} landmarks"
            ).format(len(self), n_rois, n_focuses, n_magnets, len(self.landmarks))
        )

    def save(self):
        """Saves the wafer annotations to the .magc file."""
        dlog("Saving ...")
        self.manager_to_wafer()
        config = ConfigParser.ConfigParser()

        # sections
        if self.sections:
            config.add_section(Section.string)
            config.set(Section.string, "number", str(len(self.sections)))
            for _, section in sorted(self.sections.iteritems()):
                config.add_section(section.header)
                config.set(
                    section.header, "polygon", points_to_flat_string(section.points)
                )
                config.set(
                    section.header, "center", point_to_flat_string(section.centroid)
                )
                config.set(section.header, "area", str(section.area))
                config.set(section.header, "angle", angle_to_string(section.angle))
            config.add_section("end_{}".format(Section.string))

        # focus
        focuses = [
            s.focus for _, s in sorted(self.sections.iteritems()) if s.focus is not None
        ]
        if focuses:
            config.add_section(Focus.string)
            config.set(Focus.string, "number", str(len(focuses)))
            for focus in focuses:
                config.add_section(focus.header)
                config.set(focus.header, "polygon", points_to_flat_string(focus.points))
            config.add_section("end_{}".format(Focus.string))

        # magnets
        magnets = [
            s.magnet
            for _, s in sorted(self.sections.iteritems())
            if s.magnet is not None
        ]
        if magnets:
            config.add_section(Magnet.string)
            config.set(Magnet.string, "number", str(len(magnets)))
            for magnet in magnets:
                config.add_section(magnet.header)
                config.set(
                    magnet.header, "location", point_to_flat_string(magnet.centroid)
                )
            config.add_section("end_{}".format(Magnet.string))

        # landmarks
        if self.landmarks:
            config.add_section(Landmark.string)
            config.set(Landmark.string, "number", str(len(self.landmarks)))
            for _, landmark in sorted(self.landmarks.iteritems()):
                config.add_section(landmark.header)
                config.set(
                    landmark.header, "location", point_to_flat_string(landmark.centroid)
                )
            config.add_section("end_{}".format(Landmark.string))

        # rois
        all_rois = [
            roi
            for _, section in sorted(self.sections.iteritems())
            for _, roi in sorted(section.rois.iteritems())
        ]
        if all_rois:
            config.add_section(ROI.string)
            config.set(ROI.string, "number", str(len(all_rois)))
            for roi in all_rois:
                config.add_section(roi.header)
                config.set(roi.header, "polygon", points_to_flat_string(roi.points))
                config.set(roi.header, "center", point_to_flat_string(roi.centroid))
                config.set(roi.header, "area", str(roi.area))
                config.set(roi.header, "angle", angle_to_string(roi.angle))
            config.add_section("end_{}".format(ROI.string))

        # orders
        for order_name in ("serial_order", "stage_order"):
            config.add_section(order_name)
            order = getattr(self, order_name)
            value = ",".join([str(x) for x in order]) if order else "[]"
            config.set(order_name, order_name, value)

        with open(self.magc_path, "w") as configfile:
            config.write(configfile)
        dlog("Saved to {}".format(self.magc_path))

    # ------------------------------------------------------------------
    # Manager <-> Wafer sync
    # ------------------------------------------------------------------

    def manager_to_wafer(self):
        """Populates the wafer from the roi manager.

        Typically called after the user interacted with the UI.
        """
        # Preserve transforms before clearing (needed for local-to-global conversion)
        saved_inverses = {
            section_id: section.poly_transform_inverse
            for section_id, section in self.sections.iteritems()
            if section.poly_transform_inverse is not None
        }
        self.sections.clear()
        for roi in self.manager.iterator():
            cls, section_id, annotation_id = parse_annotation_name(roi.getName())
            poly = roi
            if roi.size() <= 2:
                poly = transform_poly(poly=roi, affine=TRANSFORM_DISPLAY_INVERSE)
            if self.mode is Mode.LOCAL and section_id in saved_inverses:
                poly = transform_points_to_poly(
                    poly_to_points(poly), saved_inverses[section_id]
                )
            if cls is Section:
                self.add_section(section_id, poly)
            elif cls is ROI:
                self.sections[section_id].rois[annotation_id] = ROI(
                    section_id, annotation_id, poly
                )
            elif cls is Focus:
                self.sections[section_id].focus = Focus(section_id, poly)
            elif cls is Magnet:
                self.sections[section_id].magnet = Magnet(section_id, poly)
            elif cls is Landmark:
                self.add_landmark(section_id, poly)
        self.compute_transforms()

    def wafer_to_manager(self):
        """Draws all rois from the wafer instance into the manager."""
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
            self.manager.addRoi(landmark.poly_display)
            landmark.poly.setHandleSize(landmark.handle_size_global)
        for _, section in sorted(self.sections.iteritems()):
            for annotation in section.all_annotations():
                self.manager.addRoi(annotation.poly_display)
                annotation.poly.setHandleSize(annotation.handle_size_global)

    def wafer_to_manager_local(self):
        for id_order, section_id in enumerate(self.serial_order):
            id_manager = id_order + 1
            section = self.sections[section_id]
            transform_poly_display = AffineTransform2D()
            transform_poly_display.preConcatenate(section.poly_transform)
            transform_poly_display.preConcatenate(TRANSFORM_DISPLAY)

            for annotation in section.all_annotations():
                transform = (
                    transform_poly_display
                    if len(annotation) <= 2
                    else section.poly_transform
                )
                local_poly = transform_points_to_poly(annotation.points, transform)
                local_poly.setName(str(annotation))
                local_poly.setStrokeColor(annotation.color)
                local_poly.setImage(self.image)
                local_poly.setPosition(0, id_manager, 0)
                local_poly.setHandleSize(annotation.handle_size_local)
                self.manager.addRoi(local_poly)

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def close_mode(self):
        """Closes the current mode, saving and restoring IJ listeners."""
        self.save()
        if self.mode is Mode.GLOBAL:
            self.image.hide()
        else:
            self.image.close()
        self.manager.reset()
        for listener in self.initial_ij_key_listeners:
            IJ.getInstance().addKeyListener(listener)

    def start_global_mode(self):
        self.mode = Mode.GLOBAL
        IJ.run("Labels...", "color=white font=10 use draw")
        self.image.show()
        self.arrange_windows()
        self.wafer_to_manager()
        self.set_listeners()
        time.sleep(0.1)
        self.manager.runCommand("Show All")
        IJ.run("Labels...", "color=white font=10 use draw")
        self.manager.runCommand("Show All without labels")

    def start_local_mode(self):
        dlog("Starting local mode ...")
        self.mode = Mode.LOCAL
        self.compute_transforms()
        self.create_local_stack()
        self.wafer_to_manager()
        self.set_listeners()
        self.manager.runCommand("UseNames", "false")
        self.manager.runCommand("Show None")
        set_roi_and_update_roi_manager(0)
        self.arrange_windows()

    def rebuild_local_stack(self):
        """Closes and rebuilds the local stack image to reflect current section transforms."""
        self.image.close()
        self.create_local_stack()
        self.set_listeners()
        self.arrange_windows()

    def arrange_windows(self):
        IJ.getInstance().setLocation(
            0, IJ.getScreenSize().height - IJ.getInstance().getSize().height
        )
        window = self.image.getWindow()
        if self.image.getNSlices() > 1:
            window.remove(window.getComponents()[1])  # remove slider bar
        window.maximize()
        window.setLocation(
            IJ.getScreenSize().width
            - self.manager.getSize().width
            - window.getSize().width,
            0,
        )
        self.manager.toFront()
        time.sleep(0.05)
        IJ.selectWindow(self.image.getTitle())

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def compute_transforms(self):
        """Computes and stores transforms on each Section."""
        _, _, display_size, _ = self.get_display_parameters()
        self.local_display_size = display_size
        for _, section in self.sections.iteritems():
            transform_section_bare = AffineTransform2D()
            transform_section_bare.translate([-v for v in section.centroid])
            transform_section_bare.rotate(section.angle * Math.PI / 180.0)

            section.transform = change_basis(
                A=transform_section_bare,
                B=TRANSFORM_SUBPIXEL,
                B_inverse=TRANSFORM_SUBPIXEL_INVERSE,
            )

            translation_poly = AffineTransform2D()
            translation_poly.translate(
                [0.5 * self.local_display_size[0], 0.5 * self.local_display_size[1]]
            )
            transform_poly = transform_section_bare.copy()
            transform_poly.preConcatenate(translation_poly)
            section.poly_transform = transform_poly
            section.poly_transform_inverse = transform_poly.inverse()

    def export_transforms(self):
        dlog("Saving transforms...")
        if not any(s.transform is not None for s in self.sections.values()):
            self.compute_transforms()
        folder_transforms = mkdir_p(os.path.join(self.root, "section_transforms"))
        for section_id, section in sorted(self.sections.iteritems()):
            path_transform = os.path.join(
                folder_transforms, "transform_{:04}.txt".format(section_id)
            )
            with open(path_transform, "w") as f:
                f.write(section.transform.toString())
        dlog("Saved transforms.")

    def reorder(self, annotation_types):
        MagReorderer(self).reorder(annotation_types)

    def export_global_overview(self):
        """Saves a flattened global overview image with section labels."""
        for roi in self.manager.iterator():
            cls, section_id, _ = parse_annotation_name(roi.getName())
            if cls is Section:
                roi.setName(str(section_id))
                roi.setStrokeWidth(8)
            else:
                roi.setName("")
                roi.setStrokeWidth(1)
        IJ.run(
            "Labels...",
            (
                "color=white font="
                + str(int(self.image.getWidth() / 400.0))
                + " show use draw bold"
            ),
        )
        flattened = self.image.flatten()
        flattened_path = os.path.join(self.root, "overview_global.jpg")
        IJ.save(flattened, flattened_path)
        dlog("Flattened global image saved to {}".format(flattened_path))
        flattened.close()
        self.start_global_mode()

    def export_local_overview(self):
        """Saves an annotated stack and a local montage overview."""
        channel = ask_export_channel()
        downsampling_factor = get_downsampling_factor(self)
        highres_roi_size = compute_highres_roi_size(self, downsampling_factor)
        export_highres(
            self,
            (Section, Focus, Magnet),
            get_image_paths(self.root)[-1],
            downsampling_factor,
            highres_roi_size,
            channel,
            mkdir_p(os.path.join(self.root, "export_high_res")),
        )
        montageMaker = MontageMaker()
        stack = self.image.getStack()
        n_slices = self.image.getNSlices()
        n_rows, n_cols = get_square_row_col(n_slices)
        width = self.image.getWidth()
        height = stack.getHeight()

        montage_factor = (
            1 if width < LOCAL_SIZE_STANDARD else LOCAL_SIZE_STANDARD / float(width)
        )
        if width < LOCAL_SIZE_STANDARD:
            handle_size = 5
            stroke_size = 3
        else:
            handle_size = 1 * intr(width / LOCAL_SIZE_STANDARD)
            stroke_size = 1 * width / LOCAL_SIZE_STANDARD

        flattened_ims = []
        for id_serial, id_section in enumerate(self.serial_order):
            title = "section_magc_{:04}_serial_{:04}".format(id_section, id_serial)
            im_p = stack.getProcessor(id_serial + 1).duplicate()
            bare = ImagePlus(title, im_p)
            flattened = ImagePlus("", im_p)
            for annotation in self.manager.iterator():
                roi_cls, roi_section_id, _ = parse_annotation_name(annotation.getName())
                if roi_cls is ROI and roi_section_id == id_section:
                    cloned_roi = annotation.clone()
                    cloned_roi.setHandleSize(handle_size)
                    cloned_roi.setStrokeWidth(stroke_size)
                    flattened.setRoi(cloned_roi)
                    flattened = flattened.flatten()
            flattened.setTitle(title)
            flattened_ims.append(flattened)
            flattened_ims.append(bare)

        flattened_stack = ImageStack(width, height)
        for flattened in flattened_ims:
            flattened_stack.addSlice(flattened.getTitle(), flattened.getProcessor())
        im_flattened_stack = ImagePlus("annotated_stack", flattened_stack)
        path_annotated_stack = os.path.join(self.root, "annotated_stack.tif")
        IJ.save(im_flattened_stack, path_annotated_stack)
        im_flattened_stack.close()

        for n in range(len(self), 1, -2):
            flattened_stack.deleteSlice(n)
        im_flattened_stack = ImagePlus("flattened_stack", flattened_stack)
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
        path_overview_local = os.path.join(self.root, "overview_local.jpg")
        IJ.save(montage, path_overview_local)
        del flattened_ims
        montage.close()
        im_flattened_stack.close()
        dlog("Flattened local image saved to {}".format(path_overview_local))

    # ------------------------------------------------------------------
    # Stack creation
    # ------------------------------------------------------------------

    def _build_imgstack(self, transforms):
        """Builds an imglib2 stack from the given per-section transforms."""
        radius_x = 0.5 * self.local_display_size[0]
        radius_y = 0.5 * self.local_display_size[1]
        min_coords = [int(-radius_x), int(-radius_y)]
        max_coords = [int(radius_x), int(radius_y)]
        if DEBUG:
            dlog("local stack {} {}".format(min_coords, max_coords))
        imgs = [
            Views.interval(
                RV.transform(
                    Views.interpolate(
                        Views.extendZero(self.img_global),
                        NLinearInterpolatorFactory(),
                    ),
                    transforms[section_id],
                ),
                min_coords,
                max_coords,
            )
            for section_id in self.serial_order
        ]
        return Views.permute(Views.addDimension(Views.stack(imgs), 0, 0), 3, 2)

    def create_local_stack(self):
        """Creates and displays the interactive local stack."""
        transforms = {
            section_id: section.transform
            for section_id, section in self.sections.iteritems()
        }
        self.img_local = self._build_imgstack(transforms)
        IL.show(self.img_local)
        self.image_local = IJ.getImage()
        if DEBUG:
            dlog(
                "image local {} - {}".format(
                    self.image_local.getWidth(), self.image_local.getHeight()
                )
            )

    def create_local_straight_stack(self):
        straight_transforms = {
            section_id: AffineTransform2D() for section_id in self.sections
        }
        for section_id, t in straight_transforms.iteritems():
            t.translate(self.sections[section_id].centroid)
            straight_transforms[section_id] = t.inverse()
        IL.show(self._build_imgstack(straight_transforms))
        im = IJ.getImage()
        im.hide()
        im = im.getStack()
        w, h = im.getWidth(), im.getHeight()

        flattened_ims = []
        for id_serial, id_section in enumerate(self.serial_order):
            section = self.sections[id_section]
            if section.magnet is None:
                continue
            title = "section_magc_{:04}_serial_{:04}".format(id_section, id_serial)
            im_p = im.getProcessor(id_serial + 1).duplicate()
            flattened = ImagePlus("", im_p)

            centroid_magnet = section.magnet.centroid
            centroid_section = section.centroid

            roi_magnet = PointRoi(
                centroid_magnet[0] - centroid_section[0] + 0.5 * w,
                centroid_magnet[1] - centroid_section[1] + 0.5 * h,
            )
            roi_magnet.setHandleSize(2)
            roi_magnet.setStrokeWidth(1)
            flattened.setRoi(roi_magnet)
            flattened = flattened.flatten()
            flattened.setTitle(title)
            bare = ImagePlus(title, im_p)
            flattened_ims.append(flattened)
            flattened_ims.append(bare)

        flattened_stack = ImageStack(w, h)
        for flattened in flattened_ims:
            flattened_stack.addSlice(flattened.getTitle(), flattened.getProcessor())
        im_flattened_stack = ImagePlus("straight_magnet", flattened_stack)
        IJ.save(
            im_flattened_stack, os.path.join(self.root, "overview_straight_magnet.tiff")
        )

    def create_global_magnet_stack(self):
        SIZE = 200
        SIZE_CIRCLE = 10  # keep it even
        folder_magnet_export = mkdir_p(os.path.join(self.root, "magnet_export"))
        path_ids_slab_stage = os.path.join(self.root, "ids_slab_stage.txt")
        with open(path_ids_slab_stage, "r") as f:
            ids_slab_stage = [int(line) for line in f.readlines()]
        ip = self.image_global.getProcessor()
        for id_enum, id_slab_stage in enumerate(ids_slab_stage):
            section = self.sections.get(id_slab_stage)
            if section is None or not section.rois or section.magnet is None:
                continue
            xy = section.magnet.centroid
            top_left = [intr(xy[0] - 0.5 * SIZE), intr(xy[1] - 0.5 * SIZE)]
            roi_magnet = Roi(top_left[0], top_left[1], SIZE, SIZE)
            ip.setRoi(roi_magnet)
            id_string = "slab_{:04}".format(id_slab_stage)
            crop = ImagePlus(id_string, ip.crop())
            circle = OvalRoi(
                int(xy[0] - top_left[0] - 0.5 * SIZE_CIRCLE),
                int(xy[1] - top_left[1] - 0.5 * SIZE_CIRCLE),
                SIZE_CIRCLE,
                SIZE_CIRCLE,
            )
            circle.setHandleSize(1)
            circle.setStrokeWidth(0.3)
            crop.setRoi(circle)
            crop = crop.flatten()
            IJ.save(
                crop,
                os.path.join(
                    folder_magnet_export, "{:04}_{}.png".format(id_enum, id_string)
                ),
            )

    # ------------------------------------------------------------------
    # Annotation editing
    # ------------------------------------------------------------------

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
        cls, section_id, annotation_id = parse_annotation_name(poly_name)
        self.save()
        section = self.sections[section_id]
        if cls in (Magnet, ROI, Focus):
            if get_OK("Delete {}?".format(poly_name)):
                delete_selected_roi()
                self.image.killRoi()
                if cls is ROI:
                    del section.rois[annotation_id]
                elif cls is Focus:
                    section.focus = None
                elif cls is Magnet:
                    section.magnet = None
                self.manager.select(get_roi_index_from_name(str(section)))
        elif cls is Section:
            section_id_manager = get_roi_index_from_name(str(section))
            linked_annotations = (
                ([section.focus] if section.focus is not None else [])
                + ([section.magnet] if section.magnet is not None else [])
                + [roi for _, roi in sorted(section.rois.iteritems())]
            )
            details = "".join("{}\n \n".format(a) for a in linked_annotations)
            message = "Delete {}?{}".format(
                poly_name,
                "\n \nIt will also delete\n \n" + details if details else "",
            )
            if not get_OK(message):
                return
            self.image.killRoi()
            for linked_annotation in linked_annotations:
                delete_roi_by_index(get_roi_index_from_name(str(linked_annotation)))
            delete_roi_by_index(section_id_manager)
            self.serial_order.remove(section_id)
            self.stage_order.remove(section_id)
            del self.sections[section_id]

            self.image.close()
            self.manager.reset()
            if len(self) == 0:
                self.start_global_mode()
            else:
                self.start_local_mode()
                if section_id_manager < self.manager.getCount():
                    set_roi_and_update_roi_manager(section_id_manager)
                else:
                    set_roi_and_update_roi_manager(self.manager.getCount() - 1)

    def update_section(self, section_id, transform, ref_section_points, ref_roi_points):
        """Updates section location from a transform (used by MagReorderer)."""
        with self.set_mode(Mode.GLOBAL):
            self.add_section(
                section_id, transform_points_to_poly(ref_section_points, transform)
            )
            self.add_roi(
                section_id,
                REORDER_ROI_ID,
                transform_points_to_poly(ref_roi_points, transform),
            )

    def renumber_sections(self):
        """Renumbers the sections to have consecutive numbers without gaps.

        0,1,4,5,7 -> 0,1,2,3,4
        """
        if not user_confirmation("renumber the sections"):
            return
        if self.mode is Mode.LOCAL:
            dlog("Closing local mode. Section renumbering will be done in global mode")
            self.close_mode()
            self.start_global_mode()
        for new_key, key in enumerate(sorted(self.sections)):
            if new_key == key:
                continue
            id_serial = self.serial_order.index(key)
            self.serial_order[id_serial] = new_key
            id_stage = self.stage_order.index(key)
            self.stage_order[id_stage] = new_key

            section = self.sections[key]
            new_section = Section(new_key, section.poly)
            new_section.rois = {
                roi_id: ROI(new_key, roi_id, roi.poly)
                for roi_id, roi in section.rois.iteritems()
            }
            if section.focus is not None:
                new_section.focus = Focus(new_key, section.focus.poly)
            if section.magnet is not None:
                new_section.magnet = Magnet(new_key, section.magnet.poly)
            self.sections[new_key] = new_section
            del self.sections[key]
        self.compute_transforms()
        self.wafer_to_manager()
        dlog("{} sections have been renumbered".format(len(self)))

    # ------------------------------------------------------------------
    # Queries / suggestions
    # ------------------------------------------------------------------

    def suggest_annotation_ids(self, annotation_class):
        if annotation_class is Section:
            return suggest_ids([s.id_ for s in self.sections.values()])
        if annotation_class is Landmark:
            return suggest_ids([landmark.id_ for landmark in self.landmarks.values()])
        return []

    def suggest_roi_ids(self, section_id):
        section = self.sections.get(section_id)
        if section is None or not section.rois:
            return [ROI.make_name(section_id, 0)]
        return [
            ROI.make_name(section_id, roi_id_suggestion)
            for roi_id_suggestion in sorted(
                set(suggest_ids(section.rois.keys())).union(set(section.rois))
            )
        ]

    def get_closest_sections(self, point):
        """Returns sections sorted by distance to point."""
        return sorted(
            self.sections.values(),
            key=lambda section: get_distance(point, section.centroid),
        )

    def get_display_parameters(self):
        """Calculates [display_size, crop_size, tissue_magnet_distance] based on section size."""
        tissue_magnet_distance = 0

        section_extent = 0
        for section in self.sections.values()[: min(5, len(self))]:
            section_extent = max(section_extent, longest_diagonal(section.points))

        has_magnets = any(s.magnet is not None for s in self.sections.values())
        width = (
            intr(DISPLAY_FACTOR * section_extent)
            if not has_magnets
            else intr(DISPLAY_FACTOR * 1.2 * section_extent)
        )
        width = width - width % 2
        display_size = [width] * 2
        display_center = [intr(0.5 * display_size[0]), intr(0.5 * display_size[1])]
        crop_size = [2 * display_size[0], 2 * display_size[1]]

        if has_magnets:
            sections_with_magnets = sorted(
                [
                    (section_id, section)
                    for section_id, section in self.sections.iteritems()
                    if section.magnet is not None
                ]
            )
            tissue_magnet_distances = [
                get_distance(s.centroid, s.magnet.centroid)
                for _, s in sections_with_magnets[: min(5, len(sections_with_magnets))]
            ]
            tissue_magnet_distance = sum(tissue_magnet_distances) / len(
                tissue_magnet_distances
            )
        return display_center, tissue_magnet_distance, display_size, crop_size

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def compute_stage_order(self):
        """Computes the stage order by solving a TSP on the section centroids."""
        sorted_section_keys = sorted(self.sections)
        center_points = [
            pFloat(*self.sections[section_key].centroid)
            for section_key in sorted_section_keys
        ]
        distances = init_mat(len(self), init_value=999999)
        for a, b in itertools.combinations_with_replacement(range(len(self)), 2):
            distances[a][b] = distances[b][a] = center_points[a].distance(
                center_points[b]
            )
        self.stage_order = [
            sorted_section_keys[o] for o in compute_tsp_order(distances, self.root)
        ]

    def get_current_id_section_local(self):
        """Gets the id of the section currently displayed in the local stack."""
        return self.serial_order[self.image.getSlice() - 1]

    def push_section(self, direction):
        """Pushes the currently displayed section backward or forward in the serial order."""
        if not user_confirmation(
            "push a section and therefore modify the serial order"
        ):
            return
        delta = -1 if direction is Neighbor.PREVIOUS else 1
        id_section = self.get_current_id_section_local()
        id_serial = self.serial_order.index(id_section)
        if (id_serial == 0 and direction is Neighbor.PREVIOUS) or (
            id_serial == len(self) - 1 and direction is Neighbor.NEXT
        ):
            dlog("Cannot push the section to {} section".format(direction))
            return
        self.serial_order[id_serial], self.serial_order[id_serial + delta] = (
            self.serial_order[id_serial + delta],
            self.serial_order[id_serial],
        )
        self.wafer_to_manager()
        self.close_mode()
        self.start_local_mode()
        set_roi_and_update_roi_manager(
            get_roi_index_from_name(str(self.sections[id_section]))
        )
        dlog("The section has been pushed to the {} section".format(direction))

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def transfer_from_source_wafer(self):
        """
        Imports annotations from a source wafer .magc file using landmarks to
        map annotations into the current target wafer.
        Only available in global mode.
        """
        wafer_target = self
        path_source = get_path(
            "Select .magc file from which annotations should be imported"
        )
        if path_source is None:
            return
        wafer_source = Wafer(magc_path=path_source)

        affine_transform = affine_t(
            [landmark[0] for landmark in wafer_source.landmarks_xy],
            [landmark[1] for landmark in wafer_source.landmarks_xy],
            [landmark[0] for landmark in wafer_target.landmarks_xy],
            [landmark[1] for landmark in wafer_target.landmarks_xy],
        )

        def apply_transform(points):
            return points_to_poly(
                xy_to_points(
                    *apply_affine_t(
                        [p[0] for p in points], [p[1] for p in points], affine_transform
                    )
                )
            )

        wafer_target.sections.clear()
        for key_section, section in sorted(wafer_source.sections.iteritems()):
            wafer_target.add_section(key_section, apply_transform(section.points))
            if section.focus is not None:
                wafer_target.add_focus(
                    key_section, apply_transform(section.focus.points)
                )
            if section.magnet is not None:
                wafer_target.add_magnet(
                    key_section, apply_transform(section.magnet.points)
                )
            for key_roi, roi in section.rois.iteritems():
                wafer_target.add_roi(key_section, key_roi, apply_transform(roi.points))

        wafer_target.stage_order = wafer_source.stage_order
        wafer_target.serial_order = wafer_source.serial_order
        wafer_target.compute_transforms()
        wafer_target.wafer_to_manager()
        dlog("Completed transfer from {} to {} ...".format(wafer_source, wafer_target))
