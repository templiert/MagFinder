import os

from ij import IJ
from java.io import File, FileInputStream
from java.lang import Exception as java_exception
from java.net import URL
from java.nio.file import Files, Paths
from java.util.zip import GZIPInputStream
from org.apache.commons.io import FileUtils

from src.utils.log import dlog


def download_unzip(url, target_path):
    dlog("Downloading TSP solver from {}".format(url))
    gz_path = os.path.join(IJ.getDirectory("plugins"), "temp.gz")
    try:
        FileUtils.copyURLToFile(URL(url), File(gz_path))
        gis = GZIPInputStream(FileInputStream(gz_path))
        try:
            Files.copy(gis, Paths.get(target_path))
        finally:
            gis.close()
    except (Exception, java_exception) as e:
        dlog("Failed to download from {} due to {}".format(url, e))
    finally:
        if os.path.isfile(gz_path):
            os.remove(gz_path)
