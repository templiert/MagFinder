from src.geometry.geometry import (
    compute_angle,
    compute_area,
    compute_centroid,
    poly_to_points,
    transform_points_to_poly,
)
from src.geometry.transforms import TRANSFORM_DISPLAY


class Annotation(object):
    string = None  # subclasses must define
    color = None  # subclasses must define
    handle_size_global = None  # subclasses must define
    handle_size_local = None  # subclasses must define

    def __init__(self, poly):
        self.id_ = None  # subclasses must set
        self.poly = poly
        self.points = poly_to_points(poly)
        self.centroid = compute_centroid(self.points)
        self.area = compute_area(self.poly)
        self.angle = compute_angle(self.poly, self.points)

    def __len__(self):
        return self.poly.size()

    @classmethod
    def make_name(cls, id_):
        return "{}-{:04}".format(cls.string, id_)

    @classmethod
    def make_header(cls, id_):
        return "{}.{:04}".format(cls.string, id_)

    def __str__(self):
        return self.make_name(self.id_)

    @property
    def header(self):
        return self.make_header(self.id_)

    def __repr__(self):
        return "{} | id {} | {:.3f} {:.3f}".format(
            self.__class__.__name__, self.id_, self.centroid[0], self.centroid[1]
        )

    @property
    def poly_display(self):
        self.poly.setName(str(self))
        self.poly.setStrokeColor(self.color)
        if len(self) > 2:
            return self.poly
        poly = transform_points_to_poly(self.points, TRANSFORM_DISPLAY)
        poly.setName(str(self))
        poly.setStrokeColor(self.color)
        return poly
