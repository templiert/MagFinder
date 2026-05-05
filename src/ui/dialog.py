import sys
import threading
import time

from fiji.util.gui import GenericDialogPlus
from ij import IJ
from ij.gui import GenericDialog

from cfg import SAFETY
from src.ui.safety import Safety
from src.utils.log import dlog


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


def user_confirmation(text):
    if SAFETY is Safety.RELAXED:
        return True
    return get_OK("Are you sure you want to {}?".format(text), "Confirmation")


def get_indexes_from_user_string(userString):
    """inspired by the substackMaker of ImageJ \n
    https://imagej.nih.gov/ij/developer/api/ij/plugin/SubstackMaker.html
    Enter a range (2-30), a range with increment (2-30-2), or a list (2,5,3)
    """
    userString = userString.replace(" ", "")
    if "," in userString and "." in userString:
        return
    if "," in userString:
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
    return checkbox.label


def ask_export_channel():
    """Asks the user for the channel to use for high-res export.

    Returns the channel index, or None for single-channel images.
    """
    gd = GenericDialogPlus("High-res export")
    gd.addCheckbox("multichannel", False)
    gd.addNumericField("channel", 0, 0)
    gd.showDialog()
    if gd.wasCanceled():
        return None
    multichannel = gd.getNextBoolean()
    channel = int(gd.getNextNumber())
    return channel if multichannel else None


def thread_focus_on_OK(button):
    time.sleep(0.1)
    try:
        button.requestFocusInWindow()
    except Exception:
        pass


def focus_on_ok(dialog):
    """Trick to preselect the OK button after a dialog has been shown"""
    ok_buttons = [
        button
        for button in list(dialog.getButtons())
        if button and button.getLabel().strip().lower() == "ok"
    ]
    try:
        ok_button = ok_buttons[0]
        threading.Thread(target=thread_focus_on_OK, args=[ok_button]).start()
    except Exception:
        pass
