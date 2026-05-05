class Metric(object):
    """Similarity metrics between pairs of images."""

    INLIER_NUMBER = "inlier"
    INLIER_DISPLACEMENT = "displacement"

    @classmethod
    def all(cls):
        return cls.INLIER_NUMBER, cls.INLIER_DISPLACEMENT
