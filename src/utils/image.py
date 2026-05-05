import importlib

import loci
import loci.plugins.BF
from ij import IJ, ImagePlus
from src.utils.log import dlog
from loci.common import Region
from mpicbg.ij.plugin import NormalizeLocalContrast

# from loci.plugins.in import ImporterOptions # fails because of the "in", use import_module instead
ImporterOptions = importlib.import_module("loci.plugins.in.ImporterOptions")

DEBUG = False


def crop_open(im_path, x, y, w, h, channel):
    """Opens only the given ROI in an image saved on disk.

    Works with OME-TIFF images that do not fit into memory
    x,y,w,h are int
    """
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(w, int)
    assert isinstance(h, int)
    if DEBUG:
        dlog("crop open: {}-{}-{}-{}".format(x, y, w, h))
    options = ImporterOptions()
    options.setColorMode(ImporterOptions.COLOR_MODE_GRAYSCALE)
    options.setCrop(True)
    options.setCropRegion(0, Region(x, y, w, h))
    options.setId(im_path)
    if channel is None:
        imps = loci.plugins.BF.openImagePlus(options)
        return imps[0]
    options.setSplitChannels(True)
    imps = loci.plugins.BF.openImagePlus(options)
    return imps[channel]


def open_subpixel_crop(im_path, x, y, w, h, channel):
    """Opens only the given ROI with subpixel accuracy. See crop_open"""
    if DEBUG:
        dlog("subpixel crop: {}-{}-{}-{}".format(x, y, w, h))
    im = crop_open(im_path, int(x), int(y), w, h, channel)
    command_translate = " ".join(
        ("x={}".format(int(x) - x), "y={}".format(int(y) - y), "interpolation=Bicubic")
    )
    IJ.run(im, "Translate...", command_translate)
    return im


def rotate(im, angle_degree):
    parts_rotate = "angle={}".format(angle_degree), "grid=1", "interpolation=Bilinear"
    IJ.run(im, "Rotate... ", " ".join(parts_rotate))


def crop(im, roi):
    ip = im.getProcessor()
    ip.setRoi(roi)
    return ImagePlus("{}_cropped".format(im.getTitle()), ip.crop())


def norm_local_contrast(im, x, y, stdev, center, stretch):
    NormalizeLocalContrast().run(im.getProcessor(), x, y, stdev, center, stretch)
    return im
