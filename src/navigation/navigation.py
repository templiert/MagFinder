from java.awt import Rectangle

from src.navigation.direction import Direction
from src.utils.numeric import intr


def move_fov(direction, wafer):
    """Moves field of view of the image"""
    im = wafer.image
    canvas = im.getCanvas()
    r = canvas.getSrcRect()
    # adjust increment depending on zoom level
    increment = intr(40 / float(canvas.getMagnification()))
    xPixelIncrement = increment * (
        -1 if direction is Direction.LEFT else 1 if direction is Direction.RIGHT else 0
    )
    yPixelIncrement = increment * (
        -1 if direction is Direction.UP else 1 if direction is Direction.DOWN else 0
    )
    new_rectangle = Rectangle(
        min(max(0, r.x + xPixelIncrement), im.getWidth() - r.width),
        min(max(0, r.y + yPixelIncrement), im.getHeight() - r.height),
        r.width,
        r.height,
    )
    canvas.setSourceRect(new_rectangle)
    im.updateAndDraw()
