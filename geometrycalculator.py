from __future__ import with_statement

import itertools

import jarray
import java
from ij.gui import PointRoi, PolygonRoi
from Jama import Matrix
from java.lang import Math
from java.util import HashSet
from mpicbg.models import Point, PointMatch, RigidModel2D
from net.imglib2.realtransform import AffineTransform2D


class GeometryCalculator(object):
    @staticmethod
    def get_distance(p_1, p_2):
        """Distance between two points"""
        return Math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)

    @classmethod
    def longest_diagonal(cls, points):
        """Longest pairwise distance among all points"""
        max_diag = 0
        for p1, p2 in itertools.combinations(points, 2):
            max_diag = max(cls.get_distance(p1, p2), max_diag)
        return max_diag

    @staticmethod
    def points_to_poly(points):
        """From list of points to Fiji PointRoi or PolygonRoi"""
        """Always use floatPlogyon.
        Roi[Point, x=677, y=678] does a round()
        But Roi[Polygon, x=577, y=577, width=201, height=201] does a int()
        It should not matter as long as we always use FloatPolygon()
        """
        if len(points) == 1:
            return PointRoi(*[float(v) for v in points[0]])
        if len(points) == 2:
            polygon_type = PolygonRoi.POLYLINE
        elif len(points) > 2:
            polygon_type = PolygonRoi.POLYGON
        return PolygonRoi(
            [float(point[0]) for point in points],
            [float(point[1]) for point in points],
            polygon_type,
        )

    @staticmethod
    def poly_to_points(poly):
        float_polygon = poly.getFloatPolygon()
        result = [[x, y] for x, y in zip(float_polygon.xpoints, float_polygon.ypoints)]
        return result

    @staticmethod
    def points_to_flat_string(points):
        """[[1,2],[3,4]] -> 1,2,3,4"""
        points_flat = []
        for point in points:
            points_flat.append(point[0])
            points_flat.append(point[1])
        points_string = ",".join([str(round(x, 6)) for x in points_flat])
        return points_string

    @staticmethod
    def point_to_flat_string(point):
        """[1,2] -> 1,2"""
        flat_string = ",".join([str(round(x, 6)) for x in point])
        return flat_string

    @staticmethod
    def points_to_xy(points):
        """[[1,2],[3,4]] -> [(1,3),(2,4)]"""
        return zip(*points)

    @staticmethod
    def xy_to_points(x, y):
        return zip(x, y)

    @staticmethod
    def transform_points(source_points, aff):
        target_points = []
        for source_point in source_points:
            target_point = jarray.array([0.0, 0.0], "d")
            aff.apply(source_point, target_point)
            target_points.append(target_point)
        return target_points

    @staticmethod
    def to_imglib2_aff(trans):
        mat_data = jarray.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], java.lang.Class.forName("[D")
        )
        trans.toMatrix(mat_data)
        imglib2_transform = AffineTransform2D()
        imglib2_transform.set(mat_data)
        return imglib2_transform

    @classmethod
    def to_translation(aff):
        """The same transform with only the translation component"""
        t = AffineTransform2D()
        t.translate([aff.get(0, 2), aff.get(1, 2)])
        return t

    @classmethod
    def transform_points_to_poly(cls, source_points, aff):
        target_points = cls.transform_points(source_points, aff)
        return cls.points_to_poly(target_points)

    @staticmethod
    def affine_t(x_in, y_in, x_out, y_out):
        """Fits an affine transform to given points"""
        X = Matrix(
            jarray.array(
                [[x, y, 1.0] for (x, y) in zip(x_in, y_in)],
                java.lang.Class.forName("[D"),
            )
        )
        Y = Matrix(
            jarray.array(
                [[x, y, 1.0] for (x, y) in zip(x_out, y_out)],
                java.lang.Class.forName("[D"),
            )
        )
        aff = X.solve(Y)
        return aff

    @staticmethod
    def apply_affine_t(x_in, y_in, aff):
        X = Matrix(
            jarray.array(
                [[x, y, 1.0] for (x, y) in zip(x_in, y_in)],
                java.lang.Class.forName("[D"),
            )
        )
        Y = X.times(aff)
        x_out = [float(y[0]) for y in Y.getArrayCopy()]
        y_out = [float(y[1]) for y in Y.getArrayCopy()]
        return x_out, y_out

    @staticmethod
    def invert_affine_t(aff):
        return aff.inverse()

    @staticmethod
    def rigid_t(x_in, y_in, x_out, y_out):
        rigidModel = RigidModel2D()
        pointMatches = HashSet()
        for x_i, y_i, x_o, y_o in zip(x_in, y_in, x_out, y_out):
            pointMatches.add(
                PointMatch(
                    Point([x_i, y_i]),
                    Point([x_o, y_o]),
                )
            )
        rigidModel.fit(pointMatches)
        return rigidModel

    @staticmethod
    def apply_rigid_t(x_in, y_in, rigid_model):
        x_out = []
        y_out = []
        for x_i, y_i in zip(x_in, y_in):
            x_o, y_o = rigid_model.apply([x_i, y_i])
            x_out.append(x_o)
            y_out.append(y_o)
        return x_out, y_out

    @classmethod
    def get_imglib2_transform_scaling(cls, transform):
        s1 = [0, 0]
        s2 = [1000, 1000]
        t1, t2 = cls.transform_points([s1, s2], transform)
        return cls.get_distance(t1, t2) / float(cls.get_distance(s1, s2))

    # @classmethod
    # def apply_t(x_in, y_in, transform):
    #     if len(source_section) == 2:
    #         compute_t = cls.rigid_t
    #         apply_t = cls.apply_rigid_t
    #     else:
    #         compute_t = cls.affine_t
    #         apply_t = cls.apply_affine_t

    @classmethod
    def propagate_points(cls, source_section, source_points, target_section):
        """Transforms points linked to a source section to a target section"""
        source_section_x, source_section_y = cls.points_to_xy(source_section)
        source_points_x, source_points_y = cls.points_to_xy(source_points)
        target_section_x, target_section_y = cls.points_to_xy(target_section)
        if len(source_section) == 2:
            compute_t = cls.rigid_t
            apply_t = cls.apply_rigid_t
        else:
            compute_t = cls.affine_t
            apply_t = cls.apply_affine_t
        trans = compute_t(
            source_section_x, source_section_y, target_section_x, target_section_y
        )
        target_points_x, target_points_y = apply_t(
            source_points_x, source_points_y, trans
        )
        target_points = [[x, y] for x, y in zip(target_points_x, target_points_y)]
        return target_points
