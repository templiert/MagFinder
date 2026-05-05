from ij import IJ, WindowManager
from java.awt import Button, Checkbox, Container

from src.ui.mode import Mode
from src.utils.roimanager import get_roi_manager


def _replace_key_listeners(comp, key_listener):
    for listener in comp.getKeyListeners():
        comp.removeKeyListener(listener)
    comp.addKeyListener(key_listener)


def _replace_mouse_wheel_listeners(comp, mouse_listener):
    for listener in comp.getMouseWheelListeners():
        comp.removeMouseWheelListener(listener)
    comp.addMouseWheelListener(mouse_listener)


def _get_all_elements():
    return [
        IJ.getImage().getWindow(),
        IJ.getImage().getWindow().getCanvas(),
    ] + list(WindowManager.getAllNonImageWindows())


def add_key_listener_everywhere(key_listener, mode):
    for element in _get_all_elements():
        _replace_key_listeners(element, key_listener)

    for comp0 in get_roi_manager().getComponents():
        _replace_key_listeners(comp0, key_listener)
        for comp1 in comp0.getComponents():
            if isinstance(comp1, Button):
                comp0.remove(comp1)
            elif isinstance(comp1, Checkbox) and (mode is Mode.LOCAL):
                comp0.remove(comp1)
            else:
                _replace_key_listeners(comp1, key_listener)
                if isinstance(comp1, Container):
                    for comp2 in comp1.getComponents():
                        _replace_key_listeners(comp2, key_listener)


def add_mouse_wheel_listener_everywhere(mouse_listener):
    for element in _get_all_elements():
        _replace_mouse_wheel_listeners(element, mouse_listener)

    for comp0 in get_roi_manager().getComponents():
        _replace_mouse_wheel_listeners(comp0, mouse_listener)
        for comp1 in comp0.getComponents():
            if isinstance(comp1, Button):
                comp0.remove(comp1)
            else:
                _replace_mouse_wheel_listeners(comp1, mouse_listener)
                if isinstance(comp1, Container):
                    for comp2 in comp1.getComponents():
                        _replace_mouse_wheel_listeners(comp2, mouse_listener)
