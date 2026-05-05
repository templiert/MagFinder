from ij import IJ
from java.awt.event import MouseAdapter

from src.navigation.direction import Direction
from src.navigation.navigation import move_fov
from src.ui.mode import Mode
from src.utils.roimanager import move_roi_manager_selection


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
        elif not mouseWheelEvent.isControlDown():
            if mouseWheelEvent.getWheelRotation() == 1:
                move_fov(Direction.DOWN, self.wafer)
            else:
                move_fov(Direction.UP, self.wafer)
        elif mouseWheelEvent.isControlDown():
            if mouseWheelEvent.getWheelRotation() == 1:
                IJ.run("Out [-]")
            elif mouseWheelEvent.getWheelRotation() == -1:
                IJ.run("In [+]")

    def handle_mouse_wheel_local(self, mouseWheelEvent):
        mouseWheelEvent.consume()
        step = 10 if mouseWheelEvent.isControlDown() else 1
        move_roi_manager_selection(
            step * mouseWheelEvent.getWheelRotation(), wafer=self.wafer
        )
