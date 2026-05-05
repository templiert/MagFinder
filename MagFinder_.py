import os
import sys

from ij import IJ

sys.path.append(os.path.join(IJ.getDirectory("plugins"), "mag_finder"))

from src.wafer.wafer import Wafer  # noqa: E402

if __name__ == "__main__":
    wafer = Wafer()
    wafer.start_global_mode()
