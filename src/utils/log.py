import time
from contextlib import contextmanager

from ij import IJ


def dlog(x):
    """Double log to print and log"""
    IJ.log(x)
    print(x)


@contextmanager
def log_block(text):
    """Context manager that logs start/end of a block and its duration.

    with log_block("coarse sift matching"):
        ...
    """
    dlog(">>> {} ...".format(text))
    start = time.clock()
    try:
        yield
    finally:
        dlog("<<< {} | {:.2f}s".format(text, time.clock() - start))
