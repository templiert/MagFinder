import cv2 as cv
import os
import pathlib
import numpy as np
import pickle
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools
import json
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn import metrics
import copy
import base64
import logging

log = logging.getLogger(__name__ + ".WaferMerge")
log.setLevel(logging.DEBUG)


class WaferMerge:
    """ Wafer merge


    Parameters
    ----------
    im: np.ndarray
        the original wafer image
    template_stats: TemplateStats
        will be used for the filtering, clustering and populating the absolute templates on the final output
    candidates: list
        list of candidates discovered in the inference part
    """

    def __init__(self, im, template_stats, candidates):
        self.im = im
        self.template_stats = template_stats
        self.candidates = candidates


    def finding_best_candidates(self, area_threshold, min_distance_threshold, ratio):
        """ Find the best candidates for the output

        Steps of the method are:
        - filter tissue candidates
        - filter magnet candidates
        - pair them
        - populate the final output with the absolute templates

        Parameters
        ----------
        area_threshold: float
            goes from 0 to 1 and will be combined with one of the values from TemplateStats describing the area 
            size of the class representative
        min_distance_threshold: float
            goes from 0 to 1 and will be combined with one of the values from TemplateStats describing the min
            distance of the same class representatives
        ratio: float
            goes from 0 to 1 and represents how the image and its labels will be resized back to the original size        
        
        Returns
        -------
        data: dict
            this dictionary has the same structure as the labelme JSON file
        """
        clustered_candidates = {}
        clustered_candidates["tissue"] = self._filtering(mean_area=self.template_stats.mean_area_tissue, area_threshold=area_threshold, 
                                                    min_distance=self.template_stats.min_distance_tissue, min_distance_threshold=min_distance_threshold,  
                                                    label=2)
        clustered_candidates["mag"] = self._filtering(mean_area=self.template_stats.mean_area_magnet, area_threshold=area_threshold, 
                                                    min_distance=self.template_stats.min_distance_magnet, min_distance_threshold=min_distance_threshold,  
                                                    label=3)

        sections, centroids = self._sections_and_centroids_after_pairing(clustered_candidates, mean_tm_distance=self.template_stats.mean_distance_tissue_magnet)

        data = self._populate_with_templates(sections, centroids, 
                                            template_t=(self.template_stats.absolute_template_tissue['x'], self.template_stats.absolute_template_tissue['y']), 
                                            template_m=(self.template_stats.absolute_template_magnet['x'], self.template_stats.absolute_template_magnet['y']),
                                            ratio=ratio)
        
        log.info(f"Number of found sections is {len(sections)}")
        return data

    def save_to_file(self, data, file_name):
        """ Save the data into the JSON file

        Parameters
        ----------
        data: dict
            final output that will be stored in JSON labelme file format
        file_name: str
            path where the result will be stored
        """
        print(f"Output JSON path: {file_name}")
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)


    def _filter_by_area_size(self, areas, mean_area, threshold):
        """ Filter the candidates by the area size
        
        Parameters
        ----------
        areas: list
            list of areas of the candidates
        mean_area: int
            mean area of the class (tissue or magnet) based on the template stats
        threshold: float
            goes from 0 to 1, and represents the threshold for accepting the classes within that threshold

        Returns 
        -------
        indices: np.ndarray
            indices of candidates after area size filtering
        """
        indices = np.where(np.abs(areas - mean_area) < threshold * mean_area)[0]
        return indices

    def _filter_by_clustering(self, centroids, areas, min_distance, threshold):
        """ Clustering the candidates
        
        Parameters
        ----------
        centroids: list
            list of centroids of the candidates
        areas: list
            list of areas of the candidates
        min_distance: int
            min distance of the class (tissue or magnet) based on the template stats
        threshold: float
            goes from 0 to 1, and represents the threshold for clustering the classes within that threshold

        Returns 
        -------
        indices: np.ndarray
            indices of candidates after clustering
        """
        db = DBSCAN(eps=threshold * min_distance, min_samples=1).fit(centroids)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        log.debug("Index of the best candidate for each cluster")
        indices = [] 
        for k in set(db.labels_):
            log.debug(f"If k is negative, it is noise, in our case k is {k}")
            if k != -1:
                coreClusterMask = (db.labels_ == k) & core_samples_mask
                clusterScores = areas * coreClusterMask
                
                # TODO: check the number, can implement more similar to mean area
                log.debug("Take the 60th percentile")
                p = np.percentile(clusterScores[np.nonzero(clusterScores)], 60)
                idx_best = (np.abs(clusterScores - p)).argmin()
                indices.append(idx_best)

        return indices

    def _filtering(self, mean_area, area_threshold, min_distance, min_distance_threshold, label):
        """ Filtering of the candidates

        This filtering includes:
        - filtering by area size of the candidate
        - clustering the candidates

        Parameters
        ----------
        mean_area: int 
            mean area of the class (tissue or magnet) collected from template stats
        area_threshold: float
            goes from 0 to 1, and represents the threshold for accepting the classes within that threshold
        min_distance: int
            min distance of the class (tissue or magnet) based on the template stats
        min_distance_threshold: float
            goes from 0 to 1, and represents the threshold for clustering the classes within that threshold
        label: int
            id of the label used in coco annotation file (2 - tissue, 3 - magnet)

        Returns
        -------
        clustered_candidates: dict
            dictionary containing the candidates, its contours, area sizes and centroids
        """
        log.debug("Start general filtering (includes area filtering and clustering)")

        candidates = [candidate for candidate in self.candidates if candidate.label==label]
        contours = [candidate.contour for candidate in candidates]
        areas = np.array([cv.contourArea(contour) for contour in contours])
        moments = [cv.moments(candidate.contour) for candidate in candidates]
        centroids = np.array([np.array([M['m10']/M['m00'], M['m01']/M['m00']]).astype(int) if M['m00'] != 0 else np.array([0, 0]).astype(int) for M in moments]).astype(int)

        log.debug(f"Filter by area size, mean_area={mean_area}, threshold={area_threshold}")
        indices = self._filter_by_area_size(areas, mean_area=mean_area, threshold=area_threshold)

        candidates = [candidates[i] for i in indices]
        contours = [contours[i] for i in indices]
        areas = [areas[i] for i in indices]
        centroids = [centroids[i] for i in indices]

        log.debug(f"Clustering of candidates, min_distance={min_distance}, threshold={min_distance_threshold}")
        indices = self._filter_by_clustering(centroids, areas, min_distance=min_distance, threshold=min_distance_threshold)

        candidates = [candidates[i] for i in indices]
        contours = [contours[i] for i in indices]
        areas = [areas[i] for i in indices]
        centroids = [centroids[i] for i in indices]

        log.debug("Store results in the dictionary")
        clustered_candidates = {}
        clustered_candidates["candidates"] = candidates
        clustered_candidates["contours"] = contours
        clustered_candidates["centroids"] = centroids
        clustered_candidates["areas"] = areas   
        
        return clustered_candidates

    
    def _pairing(self, mean_tm_distance, pixel_diff, centroids_t, centroids_m, contours_t, contours_m, threshold):
        """ Pair tissue with the corresponding magnet

        mean_tm_distance: int 
            mean distance between tissue and magnet collected from the template stats
        pixel_diff: int
            for checking the difference in median colors of magnet and tissue (the 
            background has the biggest difference if detected as one of the class)
        centroids_t: list
            centroids of tissue
        centroids_m: list
            centroids of magnet
        contours_t: list
            contours of tissue
        contours_m: list
            contours of magnet
        threshold:
            goes from 0 to 1 and represents the threshold for accepting the pair based on their mean 
            tissue-magnet distance

        Returns
        -------
        pairs: list of lists
            list containing all pairs, one pair is represented as list of lenth 2 containing index of 
            tissue and index of tissue (in the candidates list)
        """
        
        log.debug("Start pairing")

        def _get_contour_median_color(contour):
            cimg = np.zeros_like(self.im)
            cv.drawContours(cimg, [contour], 0, color=255, thickness=-1)

            lst_intensities = []

            log.debug("Access the image pixels and create a 1D numpy array then add to list")
            pts = np.where(cimg == 255)
            lst_intensities.append(self.im[pts[0], pts[1]])

            return np.median(lst_intensities)


        pairs = []
        for idx, tissueCentroid in enumerate(centroids_t):
            distances = cdist([tissueCentroid], centroids_m)[0]

            partnerIndices = np.where(np.abs(distances - mean_tm_distance) < threshold * mean_tm_distance)[0]

            if len(partnerIndices) == 1:
                if partnerIndices[0] not in set([sec[1] for sec in pairs]):
                    t_pixel = _get_contour_median_color(contours_t[idx])
                    m_pixel = _get_contour_median_color(contours_m[partnerIndices[0]])
                    if np.abs(m_pixel - t_pixel) < pixel_diff:
                        pairs.append([idx, partnerIndices[0]])

            elif len(partnerIndices) > 1:
                t_pixel = _get_contour_median_color(contours_t[idx])
                m_pixels = [_get_contour_median_color(contours_m[idx]) for idx in partnerIndices]
                newPartnerIndices = np.argmin((np.abs(m_pixels - t_pixel) < pixel_diff)|(np.abs(t_pixel - m_pixels) < pixel_diff))
                if partnerIndices[newPartnerIndices] not in set([sec[1] for sec in pairs]):
                    pairs.append([idx, partnerIndices[newPartnerIndices]])

        return pairs

    def _sections_and_centroids_after_pairing(self, clustered_candidates, mean_tm_distance):
        """ Get sections and its centroids after pairing

        Parameters
        ----------
        clustered_candidates: dict
            the dictionary that contains candidates, its centroids, contours, areas
        mean_tm_distance: int
            mean distance between tissue and magnet

        Returns
        -------
        sections: list
            list of found sections, containing candidate of tissue and candidate of magnet
        centroids: list
            list of centroids, containing the centroid of tissue and centroid of magnet
        """

        sections = []
        centroids = []

        PIXEL_DIFF = 50
        centroids_t = clustered_candidates['tissue']['centroids'] 
        centroids_m = clustered_candidates['mag']['centroids']  
        contours_t = clustered_candidates['tissue']['contours'] 
        contours_m = clustered_candidates['mag']['contours'] 
        candidates_t = clustered_candidates['tissue']['candidates'] 
        candidates_m = clustered_candidates['mag']['candidates'] 

        for threshold in [0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.5, 0.75, 0.9]:

            pairs = self._pairing(mean_tm_distance, PIXEL_DIFF, centroids_t, centroids_m, contours_t, contours_m, threshold)

            log.debug("Extract the valid pairs")
            for t,m in pairs:
                sections.append([candidates_t[t], candidates_m[m]])
                centroids.append([centroids_t[t], centroids_m[m]])

            log.debug("Get not paired indices")
            indices_t = [x for x in range(len(centroids_t)) if x not in [t for t, m in pairs]]
            indices_m = [x for x in range(len(centroids_m)) if x not in [m for t, m in pairs]] 

            centroids_t = [centroids_t[i] for i in indices_t]
            centroids_m = [centroids_m[i] for i in indices_m]
            contours_t = [contours_t[i] for i in indices_t]
            contours_m = [contours_m[i] for i in indices_m]
            candidates_t = [candidates_t[i] for i in indices_t]
            candidates_m = [candidates_m[i] for i in indices_m]
            
            log.info(f"Threshold: {threshold}")
            log.info(f'found number of pairs {len(pairs)}')
            log.info(f'mag candidates for next iter {len(candidates_m)}')
            log.info(f'tissue candidates for next iter {len(candidates_t)}')

        return sections, centroids

    def _populate_with_templates(self, sections, centroids, template_t, template_m, ratio):
        """ Get the output format suitable for final labelme JSON file

        Parameters
        ----------
        sections: list
            list of found sections, containing candidate of tissue and candidate of magnet
        centroids: list
            list of centroids, containing the centroid of tissue and centroid of magnet
        template_t: list of lists (contour)
            contour of the absolute tissue template
        template_m: list of lists (contour)
            contour of the absolute magnet template
        ratio: float
            goes from 0 to 1 and represents the resize ratio, to return original image and its labels to 
            the original size

        Returns
        -------
        data: dict
            the output format suitable for final labelme JSON file
        """
        tissue_contours = [section[0].contour for section in sections]
        tissue_centroids = [centroid[0] for centroid in centroids]

        mag_contours = [section[1].contour for section in sections]
        mag_centroids = [centroid[1] for centroid in centroids]

        log.debug("Generate json file")
        data = {}
        data['imagePath'] = "wafer.tif"
        data['shapes'] = []

        log.debug("displaying the absolute template on the candidates")
        for idx, [tissueContour, magContour] in enumerate(zip(tissue_contours, mag_contours)):
            tCentroid = tissue_centroids[idx]
            mCentroid = mag_centroids[idx]
            orientation = np.angle(tCentroid[1] - mCentroid[1] + (tCentroid[0] - mCentroid[0])*1j, deg=True)
            M = cv.getRotationMatrix2D((0, 0), orientation, 1)

            templateTissue = np.array([np.array([x,y]) for x,y in zip(*template_t)])
            templateMag = np.array([np.array([x,y]) for x,y in zip(*template_m)])

            log.debug("Apply rotation")
            transformedTemplateTissue = cv.transform(np.array([templateTissue]), M)[0] + tCentroid
            transformedTemplateMag = cv.transform(np.array([templateMag]), M)[0] + mCentroid

            shape = {}
            shape['line_color'] = [0, 255, 0, 128]
            shape['points'] = [ [int(point[0]*ratio), int(point[1]*ratio)] for point in transformedTemplateTissue]
            shape['fill_color'] = None
            shape['label'] = f'tissue-{str(idx).zfill(4)}'
            data['shapes'].append(shape)

            shape = {}
            shape['line_color'] = [255, 0, 0, 128]
            shape['points'] = [ [int(point[0]*ratio),int(point[1]*ratio)] for point in transformedTemplateMag]
            shape['fill_color'] = None
            shape['label'] = f'magnet-{str(idx).zfill(4)}'
            data['shapes'].append(shape)

        data['imageData'] = None #self.im.encode('base64') 
        data['lineColor'] = [0, 255, 0, 128]
        data['fillColor'] = [255, 0, 0, 128]
        
        return data
