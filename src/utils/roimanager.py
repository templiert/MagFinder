from ij import IJ
from ij.plugin.frame import RoiManager

from src.utils.numeric import intr
from src.wafer.parse import parse_annotation_name


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
        return RoiManager()
    return manager


def _get_scroll_pane(manager):
    return next(
        component
        for component in manager.getComponents()
        if "Scroll" in str(type(component))
    )


def roi_manager_scroll_bottom():
    manager = get_roi_manager()
    scroll_bar = _get_scroll_pane(manager).getVerticalScrollBar()
    bar_max = scroll_bar.getMaximum()
    scroll_bar.setValue(intr(1.5 * bar_max))


def move_roi_manager_selection(n, wafer, shift=False):
    manager = get_roi_manager()
    if manager.getCount() == 0:
        return
    selected_index = manager.getSelectedIndex()
    if selected_index == -1:
        roi_id = get_roi_index_from_current_slice(wafer.image.getSlice())
        if roi_id is not None:
            if shift:
                roi = manager.getRoi(roi_id)
                section_id = parse_annotation_name(roi.getName())[1]
                set_roi_and_update_roi_manager(
                    get_roi_index_from_name(str(wafer.sections[section_id]))
                )
            else:
                set_roi_and_update_roi_manager(roi_id)
    else:
        manager.runCommand("Update")
        if shift:
            roi = manager.getRoi(selected_index)
            section_id = parse_annotation_name(roi.getName())[1]
            serial_id = wafer.serial_order.index(section_id)
            new_serial_id = min(len(wafer) - 1, max(0, serial_id + n))
            new_name = str(wafer.sections[wafer.serial_order[new_serial_id]])
            set_roi_and_update_roi_manager(get_roi_index_from_name(new_name))
        else:
            set_roi_and_update_roi_manager(
                min(max(0, selected_index + n), manager.getCount() - 1)
            )


def delete_selected_roi():
    get_roi_manager().runCommand("Delete")


def delete_roi_by_index(index):
    manager = get_roi_manager()
    manager.select(index)
    manager.runCommand("Delete")


def get_roi_from_name(roi_name):
    return next(
        (
            roi
            for roi in get_roi_manager().getRoisAsArray()
            if roi.getName() == roi_name
        ),
        None,
    )


def get_roi_index_from_name(name):
    return next(
        (
            id_roi
            for id_roi, roi in enumerate(get_roi_manager().getRoisAsArray())
            if roi.getName() == name
        ),
        None,
    )


def select_roi_from_name(roi_name):
    manager = get_roi_manager()
    roi_index = next(
        (
            id_roi
            for id_roi, roi in enumerate(manager.getRoisAsArray())
            if roi.getName() == roi_name
        ),
        None,
    )
    if roi_index is not None:
        manager.select(roi_index)


def set_roi_and_update_roi_manager(roi_index, select=True):
    """Keeps the selected annotation in the middle of the manager display"""
    manager = get_roi_manager()
    n_rois = manager.getCount()
    scroll_bar = _get_scroll_pane(manager).getVerticalScrollBar()
    bar_max = scroll_bar.getMaximum()
    bar_window = scroll_bar.getVisibleAmount()
    rois_per_bar = n_rois / float(bar_max)
    rois_per_window = bar_window * rois_per_bar
    scroll_value = int((roi_index - rois_per_window / 2.0) / float(rois_per_bar))
    scroll_value = max(0, scroll_value)
    scroll_value = min(scroll_value, bar_max)
    if select:
        manager.select(roi_index)
    scroll_bar.setValue(scroll_value)


def get_roi_index_from_current_slice(slice_id):
    manager = get_roi_manager()
    for roi in manager.iterator():
        if roi.getPosition() == slice_id:
            return manager.getRoiIndex(roi)
    if manager.getCount() > 0:
        return 0
    return None


def toggle_fill(annotation_type):
    manager = get_roi_manager()
    for roi in manager.iterator():
        if annotation_type.string in roi.getName():
            roi.setFillColor(None if roi.getFillColor() else annotation_type.color)
    manager.runCommand("Show All without labels")


def toggle_labels():
    manager = get_roi_manager()
    if manager.getDrawLabels():
        manager.runCommand("Show All without labels")
    else:
        manager.runCommand("UseNames", "true")
        manager.runCommand("Show All with labels")
