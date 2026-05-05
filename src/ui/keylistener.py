from fiji.util.gui import GenericDialogPlus
from ij import IJ
from ij.gui import PointRoi, PolygonRoi
from java.awt.event import KeyAdapter, KeyEvent

from src.geometry.geometry import poly_to_points
from src.navigation.direction import Direction
from src.navigation.navigation import move_fov
from src.order.neighbor import Neighbor
from src.ui.buttonclick import ButtonClick
from src.ui.dialog import (
    annotation_name_validation_dialog,
    get_indexes_from_user_string,
    user_confirmation,
)
from src.ui.help import HELP_MSG_GLOBAL, HELP_MSG_LOCAL, MSG_DRAWN_ROI_MISSING
from src.ui.mode import Mode
from src.utils.log import dlog
from src.utils.numeric import intr
from src.utils.roimanager import (
    get_roi_index_from_name,
    get_roi_manager,
    move_roi_manager_selection,
    roi_manager_scroll_bottom,
    select_roi_from_name,
    set_roi_and_update_roi_manager,
    toggle_fill,
    toggle_labels,
)
from src.wafer.focus import Focus
from src.wafer.landmark import Landmark
from src.wafer.magnet import Magnet
from src.wafer.parse import parse_annotation_name
from src.wafer.roi import ROI
from src.wafer.section import Section


class KeyListener(KeyAdapter):
    def __init__(self, wafer):
        self.wafer = wafer
        self.manager = get_roi_manager()

    def keyPressed(self, event):
        keycode = event.getKeyCode()
        if keycode == KeyEvent.VK_ESCAPE:
            self.wafer.image.killRoi()
            return
        event.consume()
        if event.getKeyCode() == KeyEvent.VK_J and self.wafer.mode is Mode.GLOBAL:
            self.wafer.reorder((Section, Focus, Magnet))
        elif keycode == KeyEvent.VK_S:
            self.wafer.save()
        elif keycode == KeyEvent.VK_Q:
            self.wafer.manager_to_wafer()
            self.wafer.compute_stage_order()
            self.wafer.close_mode()
            self.manager.close()
        elif keycode == KeyEvent.VK_O:
            self.wafer.manager_to_wafer()
            self.wafer.compute_stage_order()
            self.wafer.save()
        elif keycode == KeyEvent.VK_EQUALS:
            IJ.run("In [+]")
        elif keycode == KeyEvent.VK_MINUS:
            IJ.run("Out [-]")
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
            toggle_fill(Section)
        if keycode == KeyEvent.VK_2:
            toggle_fill(ROI)
        if keycode == KeyEvent.VK_3:
            toggle_fill(Focus)
        if keycode == KeyEvent.VK_0:
            toggle_labels()
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
        if keycode == KeyEvent.VK_U:
            self.wafer.create_local_straight_stack()
        if keycode == KeyEvent.VK_K:
            self.wafer.transfer_from_source_wafer()
        if keycode == KeyEvent.VK_Y:
            if not user_confirmation("reverse the serial order"):
                return
            self.wafer.serial_order = self.wafer.serial_order[::-1]
        if keycode == KeyEvent.VK_B:
            self.wafer.create_global_magnet_stack()

    def handle_key_m_global(self):
        self.wafer.export_global_overview()

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
            move_roi_manager_selection(-1, self.wafer, shift=keyEvent.isShiftDown())
        if keycode == KeyEvent.VK_F:
            move_roi_manager_selection(1, self.wafer, shift=keyEvent.isShiftDown())
        if keycode == KeyEvent.VK_C:
            move_roi_manager_selection(-10, self.wafer, shift=keyEvent.isShiftDown())
        if keycode == KeyEvent.VK_V:
            move_roi_manager_selection(10, self.wafer, shift=keyEvent.isShiftDown())
        if keycode in (KeyEvent.VK_E, KeyEvent.VK_R):
            self.handle_keys_e_r(keycode, shift=keyEvent.isShiftDown())
        if keycode == KeyEvent.VK_T:
            self.wafer.close_mode()
            self.wafer.start_global_mode()
        if keycode == KeyEvent.VK_G:
            self.propagate_to_neighbor_section(
                neighbor=Neighbor.PREVIOUS if keyEvent.isShiftDown() else Neighbor.NEXT
            )
        if keycode == KeyEvent.VK_1:
            self.wafer.push_section(Neighbor.PREVIOUS)
        if keycode == KeyEvent.VK_2:
            self.wafer.push_section(Neighbor.NEXT)
        keyEvent.consume()

    def handle_keys_e_r(self, keycode, shift=False):
        selected_index = self.manager.getSelectedIndex()
        if selected_index != -1:
            self.manager.runCommand("Update")
        set_roi_and_update_roi_manager(
            0
            if keycode == KeyEvent.VK_E
            else (
                self.manager.getCount() - 1
                if not shift
                else get_roi_index_from_name(
                    str(self.wafer.sections[self.wafer.serial_order[-1]])
                )
            )
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
            section_id = self.wafer.get_current_id_section_local()
            if drawn_roi.size() == 2:
                name_suggestions.append(Magnet.make_name(section_id))
            else:
                name_suggestions.append(Section.make_name(section_id))
                name_suggestions += self.wafer.suggest_roi_ids(section_id)
            name_suggestions.append(Focus.make_name(section_id))
        elif self.wafer.mode is Mode.GLOBAL:
            closest_sections = self.wafer.get_closest_sections(
                drawn_roi.getContourCentroid()
            )
            if drawn_roi.size() == 2:
                name_suggestions += [
                    Magnet.make_name(section.id_) for section in closest_sections[:3]
                ]
                name_suggestions += [
                    Landmark.make_name(id_)
                    for id_ in self.wafer.suggest_annotation_ids(Landmark)
                ]
            else:
                name_suggestions += [
                    Section.make_name(id_)
                    for id_ in self.wafer.suggest_annotation_ids(Section)
                ]
                name_suggestions += [
                    name_suggestion
                    for section in closest_sections[:3]
                    for name_suggestion in self.wafer.suggest_roi_ids(section.id_)
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
        cls, section_id, annotation_id = parse_annotation_name(annotation_name)
        if self.wafer.mode is Mode.LOCAL and cls is Section:
            if not user_confirmation("modify the current *section*"):
                return
        drawn_roi.setHandleSize(
            cls.handle_size_local
            if self.wafer.mode is Mode.LOCAL
            else cls.handle_size_global
        )
        if cls is ROI:
            self.wafer.add_roi(section_id, annotation_id, drawn_roi)
        elif cls is Section:
            self.wafer.add_section(
                section_id, self.wafer.to_global(section_id, drawn_roi)
            )
        elif cls is Focus:
            self.wafer.add_focus(section_id, drawn_roi)
        elif cls is Magnet:
            self.wafer.add_magnet(section_id, drawn_roi)
        elif cls is Landmark:
            self.wafer.add_landmark(section_id, drawn_roi)
        drawn_roi.setStrokeColor(cls.color)
        self.wafer.image.killRoi()
        if self.wafer.mode is Mode.LOCAL and cls is Section:
            self.wafer.compute_transforms()
            self.wafer.rebuild_local_stack()
        self.wafer.wafer_to_manager()

        if self.wafer.mode is Mode.LOCAL:
            self.manager.select(get_roi_index_from_name(annotation_name))
        else:
            roi_manager_scroll_bottom()
        dlog("Annotation {} added".format(annotation_name))

    def handle_key_m_local(self):
        self.wafer.export_local_overview()

    def handle_key_x_local(self):
        self.wafer.remove_current()

    def handle_key_p_local(self):
        """Propagation tool: propagates a single annotation to a range of sections."""
        self.manager.runCommand("Update")
        self.wafer.manager_to_wafer()
        selected_indexes = self.manager.getSelectedIndexes()
        if len(selected_indexes) != 1:
            IJ.showMessage(
                "Warning", "Select only one annotation to use the propagation tool"
            )
            return
        selected_poly = self.manager.getRoi(selected_indexes[0])
        poly_name = selected_poly.getName()
        cls, section_id, annotation_id = parse_annotation_name(poly_name)
        if cls is Section:
            IJ.showMessage(
                "Info",
                "Sections cannot be propagated. Only rois, focus, magnets can be propagated",
            )
            return
        min_section_id = min(self.wafer.sections)
        max_section_id = max(self.wafer.sections)

        dialog = GenericDialogPlus("Propagation")
        dialog.addMessage(
            (
                "This {} is defined in section #{}."
                "\nTo what sections do you want to propagate this {}?"
            ).format(cls.string, section_id, cls.string)
        )
        dialog.addStringField(
            "Enter a range or single values separated by commas. "
            + "Range can be start-end (4-7 = 4,5,6,7) or "
            + "start-end-increment (2-11-3 = 2,5,8,11).",
            "{}-{}".format(min_section_id, max_section_id),
        )
        range_field = dialog.getStringFields()[0]
        dialog.addButton(
            "All sections {}-{}".format(min_section_id, max_section_id),
            ButtonClick(range_field, "{}-{}".format(min_section_id, max_section_id)),
        )
        dialog.addButton(
            "First half of the sections {}-{}".format(
                min_section_id, intr(max_section_id / 2.0)
            ),
            ButtonClick(
                range_field, "{}-{}".format(min_section_id, intr(max_section_id / 2.0))
            ),
        )
        dialog.addButton(
            "Every second section {}-{}-2".format(min_section_id, max_section_id),
            ButtonClick(range_field, "{}-{}-2".format(min_section_id, max_section_id)),
        )
        dialog.showDialog()
        if not dialog.wasOKed():
            return
        user_range = dialog.getNextString()
        input_indexes = get_indexes_from_user_string(user_range)
        dlog("User input indexes from Propagation Dialog: {}".format(input_indexes))
        valid_input_indexes = [i for i in input_indexes if i in self.wafer.sections]
        if not valid_input_indexes:
            return

        self.wafer.propagate_annotation(
            cls, section_id, annotation_id, valid_input_indexes
        )

    def propagate_to_neighbor_section(self, neighbor):
        neighbor_section_name = self.wafer.propagate_section_to_neighbor(neighbor)
        if neighbor_section_name is not None:
            select_roi_from_name(neighbor_section_name)
