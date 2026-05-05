import itertools


def pairwise(iterable):
    """from https://docs.python.org/3/library/itertools.html#itertools.pairwise
    'ABCD' --> AB BC CD"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
