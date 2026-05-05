import os

from ij import IJ
from ij.gui import PointRoi
from java.awt import Polygon as AwtPolygon
from java.awt.geom import AffineTransform
from java.lang import Math
from java.util import ArrayList, HashSet
from java.util.concurrent.atomic import AtomicInteger
from mpicbg.ij import SIFT, FeatureTransform
from mpicbg.imagefeatures import FloatArray2DSIFT
from mpicbg.models import (
    AffineModel2D,
    NotEnoughDataPointsException,
    PointMatch,
    RigidModel2D,
)

from src.geometry.geometry import change_basis, get_distance, translation_norm
from src.order.metric import Metric
from src.utils.file import roi_image_filename
from src.utils.image import crop
from src.utils.log import dlog
from src.utils.parallel import start_threads
from src.utils.serialize import deserialize_parallel, serialize, serialize_parallel

DEBUG = False
ROOT = None  # set to the roi folder path when DEBUG = True


def create_sift_parameters(fdSize, initialSigma, steps, minOctaveSize, maxOctaveSize):
    p = FloatArray2DSIFT.Param().clone()
    p.fdSize = fdSize
    p.initialSigma = initialSigma
    p.steps = steps
    p.maxOctaveSize = maxOctaveSize
    p.minOctaveSize = minOctaveSize
    return p


def sift_params_to_string(sift_params):
    return "{}_{}_{}_{}_{}_{:.1f}".format(
        sift_params.fdBins,
        sift_params.fdSize,
        sift_params.steps,
        sift_params.minOctaveSize,
        sift_params.maxOctaveSize,
        sift_params.initialSigma,
    )


def parallel_compute_sift(atom, paths, saveFolder, sift_params, roi=None):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k >= len(paths):
            break
        float_SIFT = FloatArray2DSIFT(sift_params)
        ij_SIFT = SIFT(float_SIFT)
        im = IJ.openImage(paths[k])
        if roi is not None:
            im = crop(im, roi)
        ip = im.getProcessor()
        features = HashSet()
        ij_SIFT.extractFeatures(ip, features)
        dlog("Section {}: {} features extracted".format(k, features.size()))
        im.close()
        serialize(features, os.path.join(saveFolder, "features_{:04}".format(k)))
        del features


def get_SIFT_similarity_parallel(
    atom,
    pairs,
    section_features,
    pairwise_costs,
    affine_transforms,
    transform_type,
    translation_threshold,
    highres_w,
):
    translation_center = AffineTransform()
    translation_center.translate(0.5 * highres_w, 0.5 * highres_w)
    while atom.get() < len(pairs):
        k = atom.getAndIncrement()
        if k >= len(pairs):
            break
        id1, id2 = pairs[k]
        if k % 100 == 0:
            dlog("Processing pair {} ".format((id1, id2)))
        get_SIFT_similarity(
            id1,
            id2,
            section_features[id1],
            section_features[id2],
            pairwise_costs,
            affine_transforms,
            transform_type,
            translation_threshold,
            highres_w,
            translation_center,
            0,  # attempt
        )


def get_SIFT_similarity(
    id1,
    id2,
    features_1,
    features_2,
    pairwise_costs,
    affine_transforms,
    transform_type,
    translation_threshold,
    highres_w,
    translation_center,
    attempt,
):
    """Computes SIFT similarity between two sections and records pairwise costs.

    Fits a rigid or affine model via RANSAC on matched SIFT features.
    If the resulting transform is within the translation threshold,
    then records inlier displacement and inlier number costs symmetrically.
    If the translation is too large (likely a misalignment),
    retries with features spatially filtered to a shrinking central region,
    recursing up to ~9 times before giving up.
    """
    dlog("processing pair {}".format((id1, id2)))
    highres_center = [0.5 * highres_w] * 2

    model_found, model, inliers = _fit_model(features_1, features_2, transform_type)
    if not model_found:
        dlog("no model found {}".format((id1, id2)))
        return

    center_rebased_transform = change_basis(model.createAffine(), translation_center)
    _log_match_result((id1, id2), inliers, center_rebased_transform)

    if is_translation_valid(center_rebased_transform, translation_threshold):
        affine_transforms[(id1, id2)] = model.createInverse()
        affine_transforms[(id2, id1)] = model
        _record_costs(pairwise_costs, id1, id2, inliers)
        return

    if DEBUG:
        _debug_show_inliers(id1, id2, inliers)

    filtering_radius = 0.5 * (1 - 0.1 * (attempt + 1)) * highres_w
    dlog("over translation {} filtering radius {}".format((id1, id2), filtering_radius))
    new_features_1 = filter_features(features_1, highres_center, filtering_radius)
    new_features_2 = filter_features(features_2, highres_center, filtering_radius)
    if len(new_features_2) < 5:
        dlog("less than 5 features after filtering pair {}".format((id1, id2)))
        return
    dlog("features | before {} | after {}".format(len(features_2), len(new_features_2)))
    if DEBUG:
        _debug_show_filtered_features(id2, new_features_2)
    get_SIFT_similarity(
        id1,
        id2,
        new_features_1,
        new_features_2,
        pairwise_costs,
        affine_transforms,
        transform_type,
        translation_threshold,
        highres_w,
        translation_center,
        attempt + 1,
    )


def _log_match_result(pair, inliers, center_rebased_transform):
    msg = (
        "model found in section pair {} | distance {:.1f} | {} inliers"
        " | center_rebased_transform {} | translation norm {}"
    )
    dlog(
        msg.format(
            pair,
            100 * PointMatch.meanDistance(inliers),
            len(inliers),
            center_rebased_transform,
            translation_norm(center_rebased_transform),
        )
    )


def _debug_show_inliers(id1, id2, inliers):
    """Shows inlier point ROIs on the section images. Requires DEBUG = True and ROOT set."""
    p1, p2 = inliers_to_polygons(inliers)
    for p, id_ in zip((p1, p2), (id1, id2)):
        im = IJ.openImage(os.path.join(ROOT, roi_image_filename(id_)))
        im.show()
        im.setRoi(PointRoi(p), True)
        im.setTitle("{}_{}".format((id1, id2), id_))


def _debug_show_filtered_features(id2, new_features_2):
    """Shows filtered feature point ROI on section image. Requires DEBUG = True and ROOT set."""
    new_features_pointroi = PointRoi(features_to_polygon(new_features_2))
    dlog("A".center(100, "-"))
    im2 = IJ.openImage(os.path.join(ROOT, roi_image_filename(id2)))
    im2.show()
    im2.setRoi(new_features_pointroi, True)


def inliers_to_polygons(inliers):
    """For debugging purposes"""
    p1, p2 = AwtPolygon(), AwtPolygon()
    for inlier in inliers:
        p1.addPoint(*(int(coord) for coord in inlier.getP1().getL()))
        p2.addPoint(*(int(coord) for coord in inlier.getP2().getL()))
    return p1, p2


def features_to_polygon(features):
    """For debugging purposes"""
    p = AwtPolygon()
    for feature in features:
        p.addPoint(*(int(coord) for coord in feature.location))
    return p


def filter_features(features, center, radius):
    """Returns features located within radius distance of center."""
    filtered_features = HashSet()
    for feature in features:
        if get_distance(feature.location, center) < radius:
            filtered_features.add(feature)
    return filtered_features


def is_translation_valid(affine, translation_threshold):
    """True if the norm of the translation is below a threshold.

    affine_transform is a AffineModel2D (not AffineTransform2D)
    """
    return (
        Math.sqrt(affine.getTranslateX() ** 2 + affine.getTranslateY() ** 2)
        < translation_threshold
    )


def _fit_model(features_1, features_2, transform_type):
    """Matches features and fits a model via RANSAC.

    Returns (model_found, model, inliers).
    """
    candidates = ArrayList()
    FeatureTransform.matchFeatures(features_1, features_2, candidates, 0.92)
    inliers = ArrayList()
    model = RigidModel2D() if transform_type == "rigid" else AffineModel2D()
    try:
        model_found = model.filterRansac(
            candidates,
            inliers,
            1000,  # iterations
            20,  # maxDisplacement
            0.001,  # ratioOfConservedFeatures
        )
    except NotEnoughDataPointsException:
        model_found = False
    return model_found, model, inliers


def _record_costs(pairwise_costs, id1, id2, inliers):
    """Records inlier displacement and inlier number costs symmetrically."""
    inlier_displacement = 100 * PointMatch.meanDistance(inliers)
    inlier_number_cost = 1000 / float(len(inliers))
    pairwise_costs[Metric.INLIER_DISPLACEMENT][id1][id2] = inlier_displacement
    pairwise_costs[Metric.INLIER_DISPLACEMENT][id2][id1] = inlier_displacement
    pairwise_costs[Metric.INLIER_NUMBER][id1][id2] = inlier_number_cost
    pairwise_costs[Metric.INLIER_NUMBER][id2][id1] = inlier_number_cost


def serialize_matchings(costs, affine_transforms, folder, percentage_cores):
    n_sections = len(costs.values()[0])
    to_serialize = [
        [
            [costs[metric][i] for metric in Metric.all()],
            {
                (i, j): affine_transforms[(i, j)]
                for j in range(n_sections)
                if (i, j) in affine_transforms
            },
        ]
        for i in range(n_sections)
    ]
    start_threads(
        serialize_parallel,
        fraction_cores=0.01 * percentage_cores,
        arguments=(
            AtomicInteger(0),
            to_serialize,
            [
                os.path.join(folder, "matches_section_{:04}".format(i))
                for i in range(n_sections)
            ],
        ),
    )


def deserialize_matchings(folder, percentage_cores):
    filenames = sorted(os.listdir(folder))
    deserialized = [None] * len(filenames)
    start_threads(
        deserialize_parallel,
        fraction_cores=0.01 * percentage_cores,
        arguments=(
            AtomicInteger(0),
            [os.path.join(folder, filename) for filename in filenames],
            deserialized,
        ),
    )
    costs = {metric: [] for metric in Metric.all()}
    affine_transforms = {}
    for section_costs, section_affine_transforms in deserialized:
        for metric, cost in zip(Metric.all(), section_costs):
            costs[metric].append(cost)
        affine_transforms.update(section_affine_transforms)
    return costs, affine_transforms
