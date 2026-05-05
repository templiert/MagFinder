"""
Coordinate convention:
Top-left of pixel is (0  , 0  )
Center of pixel is   (0.5, 0.5)
"""

import itertools

import jarray
import java
from ij.gui import PointRoi, PolygonRoi
from Jama import Matrix
from java.lang import Math
from java.util import HashSet
from mpicbg.models import Point, PointMatch, RigidModel2D
from net.imglib2.realtransform import AffineTransform2D

TWO_D_DOUBLE_ARRAY = java.lang.Class.forName("[D")


def get_distance(p_1, p_2):
    """Distance between two points."""
    return Math.sqrt((p_2[0] - p_1[0]) ** 2 + (p_2[1] - p_1[1]) ** 2)


def longest_diagonal(points):
    """Longest pairwise distance among all points."""
    return max(get_distance(p1, p2) for p1, p2 in itertools.combinations(points, 2))


def points_to_poly(points):
    """From list of points to Fiji PointRoi or PolygonRoi.

    Roi[Point, x=677, y=678] does a round()
    But Roi[Polygon, x=577, y=577, width=201, height=201] does a int()
    It should not matter as long as we always use FloatPolygon()
    """
    if len(points) == 1:
        return PointRoi(*[float(v) for v in points[0]])
    if len(points) == 2:
        polygon_type = PolygonRoi.POLYLINE
    else:
        polygon_type = PolygonRoi.POLYGON
    return PolygonRoi(
        [float(point[0]) for point in points],
        [float(point[1]) for point in points],
        polygon_type,
    )


def poly_to_points(poly):
    """Polygon to points."""
    float_polygon = poly.getFloatPolygon()
    return list(zip(float_polygon.xpoints, float_polygon.ypoints))


def transform_poly(poly, affine):
    """Transforms polygon with affine transform."""
    poly_transformed = transform_points_to_poly(poly_to_points(poly), affine)
    poly_transformed.setName(poly.getName())
    return poly_transformed


def points_to_flat_string(points):
    """[[1,2],[3,4]] -> 1,2,3,4"""
    return ",".join([str(round(value, 6)) for point in points for value in point])


def point_to_flat_string(point):
    """[1,2] -> 1,2"""
    return ",".join([str(round(x, 6)) for x in point])


def points_to_xy(points):
    """[[1,2],[3,4],[5,6]] -> [(1,3,5),(2,4,6)]"""
    return zip(*points)


def xy_to_points(x, y):
    return zip(x, y)


def transform_points(source_points, affine):
    target_points = []
    for source_point in source_points:
        target_point = jarray.array([0.0, 0.0], "d")
        affine.apply(source_point, target_point)
        target_points.append(target_point)
    return target_points


def to_imglib2_aff(trans):
    matrix = jarray.array([[0.0] * 3, [0.0] * 3], TWO_D_DOUBLE_ARRAY)
    trans.toMatrix(matrix)
    imglib2_transform = AffineTransform2D()
    imglib2_transform.set(matrix)
    return imglib2_transform


def transform_points_to_poly(source_points, affine):
    return points_to_poly(transform_points(source_points, affine))


def affine_t(x_in, y_in, x_out, y_out):
    """Fits an affine transform to given points"""
    X = Matrix(
        jarray.array([[x, y, 1.0] for (x, y) in zip(x_in, y_in)], TWO_D_DOUBLE_ARRAY)
    )
    Y = Matrix(
        jarray.array([[x, y, 1.0] for (x, y) in zip(x_out, y_out)], TWO_D_DOUBLE_ARRAY)
    )
    return X.solve(Y)


def apply_affine_t(x_in, y_in, affine):
    X = Matrix(
        jarray.array([[x, y, 1.0] for (x, y) in zip(x_in, y_in)], TWO_D_DOUBLE_ARRAY)
    )
    Y = X.times(affine)
    x_out = [float(y[0]) for y in Y.getArrayCopy()]
    y_out = [float(y[1]) for y in Y.getArrayCopy()]
    return x_out, y_out


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


def apply_rigid_t(x_in, y_in, rigid_model):
    x_out = []
    y_out = []
    for x_i, y_i in zip(x_in, y_in):
        x_o, y_o = rigid_model.apply([x_i, y_i])
        x_out.append(x_o)
        y_out.append(y_o)
    return x_out, y_out


def get_imglib2_transform_scaling(transform):
    s1 = [0, 0]
    s2 = [1000, 1000]
    t1, t2 = transform_points([s1, s2], transform)
    return get_distance(t1, t2) / float(get_distance(s1, s2))


def propagate_points(source_section, source_points, target_section):
    """Transforms points linked to a source section to a target section."""
    source_section_x, source_section_y = points_to_xy(source_section)
    source_points_x, source_points_y = points_to_xy(source_points)
    target_section_x, target_section_y = points_to_xy(target_section)
    if len(source_section) == 2:
        compute, apply = rigid_t, apply_rigid_t
    else:
        compute, apply = affine_t, apply_affine_t
    trans = compute(
        source_section_x, source_section_y, target_section_x, target_section_y
    )
    target_points_x, target_points_y = apply(source_points_x, source_points_y, trans)
    return [[x, y] for x, y in zip(target_points_x, target_points_y)]


def change_basis(A, B, B_inverse=None):
    """Change A into B basis. Does not modify A in place."""
    affine = AffineTransform2D()
    affine.concatenate(B.inverse() if B_inverse is None else B_inverse)
    affine.concatenate(A)
    affine.concatenate(B)
    return affine


def translation_norm(affine):
    """Norm of the translation vector of an affine transform."""
    return Math.sqrt(affine.getTranslateX() ** 2 + affine.getTranslateY() ** 2)


def angle_to_string(angle):
    return str(((angle - 90) % 360) - 180)


def compute_centroid(points):
    """Barycenter of the contour vertices."""
    if len(points) == 1:
        return list(points[0])
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return [x_sum / float(len(points)), y_sum / float(len(points))]


def compute_area(poly):
    """Bounding box area."""
    bounds = poly.getBounds()
    return bounds.getHeight() * bounds.getWidth()


def compute_angle(poly, points):
    if len(points) < 2:
        return None
    return poly.getFloatAngle(points[0][0], points[0][1], points[1][0], points[1][1])
