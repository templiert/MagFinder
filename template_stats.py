import numpy as np
import cv2 as cv
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from misc import getCentroid
import json
import matplotlib.pyplot as plt


class TemplateStats:

    def __init__(self, absolute_template_tissue=None, absolute_template_magnet=None, 
                mean_area_tissue=None, mean_area_magnet=None, mean_distance_tissue_magnet=None,
                min_distance_tissue=None, min_distance_magnet=None,
                overlap=None, distance_inside_contour=None):
        self.absolute_template_tissue = absolute_template_tissue or {'x': [], 'y': []}
        self.absolute_template_magnet = absolute_template_magnet or {'x': [], 'y': []}

        self.mean_area_tissue = mean_area_tissue or 0
        self.mean_area_magnet = mean_area_magnet or 0
        self.mean_distance_tissue_magnet = mean_distance_tissue_magnet or 0
        self.min_distance_tissue = min_distance_tissue or 0
        self.min_distance_magnet = min_distance_magnet or 0
        self.overlap = overlap or 0
        self.distance_inside_contour = distance_inside_contour or 0

    def collect(self, labelme_init, debug=False):
        sections = list(zip(*(iter(labelme_init['shapes'][:-1]),) * 3))

        # get template with least number of points
        best_section_i = 0
        min_num_points = 100
        for i, [t, m, e] in enumerate(sections):
            num_points = len(t['points']) + len(m['points'])
            if min_num_points > num_points:
                min_num_points = num_points
                best_section_i = i
        t, m, e = sections[best_section_i]
        print(f"Min number of total points of contour is: {min_num_points}")

        # get the `absolute template`
        # create the 'absolute template' from the first section template
        # this 'absolute template' is then used for display and proofreading    
        tCentroid = getCentroid(t['points'])
        mCentroid = getCentroid(m['points'])
        orientation = np.angle(tCentroid[1]-mCentroid[1] + (tCentroid[0]-mCentroid[0])*1j, deg=True)

        # rotation matrix
        M = cv.getRotationMatrix2D((0, 0), -orientation, 1)

        # apply straightening rotation and substract centroids
        tPointsStraight = cv.transform(np.array([t['points']]), M)[0]
        mPointsStraight = cv.transform(np.array([m['points']]), M)[0]

        tPointsStraight = tPointsStraight - getCentroid(tPointsStraight)
        mPointsStraight = mPointsStraight - getCentroid(mPointsStraight)

        if debug:
            plt.fill(*tPointsStraight.T, c="#ff6666", label='tissue', fill=False)
            plt.fill(*mPointsStraight.T, c="#00cccc", label='magnet', fill=False)
            plt.legend()
            plt.show()
        
        tPointsStraight = tPointsStraight.T.astype(int).tolist()
        mPointsStraight = mPointsStraight.T.astype(int).tolist()

        self.absolute_template_tissue = {'x':tPointsStraight[0], 'y':tPointsStraight[1]}
        self.absolute_template_magnet = {'x':mPointsStraight[0], 'y':mPointsStraight[1]}
        
        templateTissueAreas = []
        templateMagAreas = []
        tissueMagDistances = []
        
        tissueMinDistances = []
        magMinDistances = []

        max_overlap = 0
        max_distance_inside_contour = 0
        for [t, m, e] in sections: # tissue, magnet, envelope
            # write the areas
            templateTissueAreas.append(cv.contourArea(np.array(t['points'])))
            templateMagAreas.append(cv.contourArea(np.array(m['points'])))
            # write the tissueMagDistances
            tissueMagDistances.append(euclidean(getCentroid(t['points']), getCentroid(m['points'])))
            
            tissueMinDistances.append(2*np.min(cdist([getCentroid(t['points'])], np.array(t['points']))))
            magMinDistances.append(2*np.min(cdist([getCentroid(m['points'])], np.array(m['points']))))

            overlap = int(1.2 * np.max(cdist(e['points'], e['points'])))
            if overlap > max_overlap:
                max_overlap = overlap

            distance_inside_contour = int(np.max([int(np.max(cdist(t['points'], t['points']))), int(np.max(cdist(m['points'], m['points'])))]))
            if distance_inside_contour > max_distance_inside_contour:
                max_distance_inside_contour = distance_inside_contour

        self.distance_inside_contour = max_distance_inside_contour
        self.overlap = max_overlap
        self.mean_area_tissue = int(np.mean(templateTissueAreas))
        self.mean_area_magnet = int(np.mean(templateMagAreas))
        self.mean_distance_tissue_magnet = int(np.mean(tissueMagDistances))
        self.min_distance_tissue = int(np.mean(tissueMinDistances))
        self.min_distance_magnet = int(np.mean(magMinDistances))
             
    def save_to_file(self, file_name):
        template_info = json.dumps(self.__dict__, indent=4)

        with open(file_name, 'w') as f:
            json.dump(template_info, f)


