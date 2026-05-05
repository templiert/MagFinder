from net.imglib2.realtransform import AffineTransform2D

# Applied to <=2 point annotations when going wafer -> manager,
# because Fiji displays them with a (0.5, 0.5) offset (different subpixel convention).
TRANSFORM_DISPLAY = AffineTransform2D()
TRANSFORM_DISPLAY.translate([-0.5, -0.5])

# Applied when going manager -> wafer (inverse of TRANSFORM_DISPLAY).
TRANSFORM_DISPLAY_INVERSE = TRANSFORM_DISPLAY.inverse()

# Fixes the discrepancy between the Fiji ROI convention (top-left of pixel)
# and the ImgLib2 convention (center of pixel).
TRANSFORM_SUBPIXEL = AffineTransform2D()
TRANSFORM_SUBPIXEL.translate([0.5, 0.5])
TRANSFORM_SUBPIXEL_INVERSE = TRANSFORM_SUBPIXEL.inverse()
