import re
import os
import sys
import pathlib
import json
import pandas as pd
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import yaml
import logging

logging.basicConfig(filename='log_pipeline.txt', filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)

def getCentroid(contour):
    """ Calculate the centroid of the given contour

    Parameters
    ----------
    contour: np.array
        Contour from which we want to get the centroid

    Returns
    -------
    : np.array
        (x, y) coordinates of the centroid of the contour
    """
    M = cv.moments(np.array(contour))
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return np.array([x,y])



def get_config(file_name, printing=False):
    # check the YAML
    def _pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key), end="")
            if isinstance(value, dict):
                print()
                _pretty(value, indent+1)
            else:
                print(': ' + str(value))

    with open(file_name, 'r') as f:
        configurations = yaml.load(f, Loader=yaml.FullLoader)
    
    if printing:
        _pretty(configurations)
    
    return configurations

def set_config(configurations, file_name, printing=False):
    if printing:
        pretty(configurations)
    
    with open(file_name, 'w') as f:
        yaml.dump(configurations, f, default_flow_style=False)
    
####input_config = get_config(os.path.join(WAFER_DATA_PATH, "input_data_config.yaml"), printing=True)

def read_input_data(labelme_template_path=None, original_image_path=None, fluo_image_path=None):
    if labelme_template_path:
        with open(labelme_template_path, 'r') as f:
            labelme_init = json.load(f)
    else:
        labelme_init = None
        
    if original_image_path:
        if not os.path.exists(original_image_path):
            raise Exception(f"There is no image file called {original_image_path}. Please put it in this location.")
        im = cv.imread(original_image_path, 0)
    else:
        im = None

    if fluo_image_path:
        if not os.path.exists(fluo_image_path):
            raise Exception(f"There is no image file called {fluo_image_path}. Please put it in this location.")
        im_fluo = cv.imread(fluo_image_path, 0)
    else:
        im_fluo = None

    return labelme_init, im, im_fluo

def draw_labels(image, labels_df, thickness, color, fill_poly):
    for row in range(labels_df.shape[0]):
        pts = np.array(labels_df.loc[row])
        pts = pts[~np.isnan(pts)].reshape((-1,1,2))

        if fill_poly:
            image = cv.fillPoly(image, np.int32([pts]), color=rgb(color))
        else:
            image = cv.polylines(image, np.int32([pts]), True, color=compute_rgb(color), thickness=thickness)
    
    return image

def compute_rgb(hex_color):
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))

def labelme_plot(labelme_json, num_sections, extrct_img):
    
    with open(labelme_json, 'r') as f:
        data = json.load(f)

    thickness = 5
    magnet_color = "#00cccc"
    tissue_color = "#ff6666"

    fill_poly = False

    tissues = []
    magnets = []
    for polygon in data["shapes"]:  
        if polygon["label"].startswith("tissue") or polygon["label"] == "b1":
            tissues.append(np.asarray(polygon["points"]).flatten())           
            
        if polygon["label"].startswith("magnet") or polygon["label"] == "m1":
            magnets.append(np.asarray(polygon["points"]).flatten())

    seg_coord_tissues = pd.DataFrame(tissues)
    print(f"Number of found tissue parts: {seg_coord_tissues.shape[0]}")
    
    #display(seg_coord_tissues)
    
    seg_coord_mag = pd.DataFrame(magnets)
    print(f"Number of found magnetic parts: {seg_coord_mag.shape[0]}")

    if num_sections != 0:
        print(f"Labeling rate: {seg_coord_tissues.shape[0]}/{num_sections} ({seg_coord_tissues.shape[0]/num_sections*100:.2f} %)")
    else:
        print(f"Labeling rate: {seg_coord_tissues.shape[0]})")
    
    extrct_img = draw_labels(extrct_img, seg_coord_mag, thickness, magnet_color, fill_poly)
    extrct_img = draw_labels(extrct_img, seg_coord_tissues, thickness, tissue_color, fill_poly)

    legend_elements = [Line2D([0], [0], color=magnet_color, lw=thickness, label='Magnet Part'),
                    Line2D([0], [0], color=tissue_color, lw=thickness, label='Brain Part')]


    plt.figure(figsize=(40, 40))
    plt.imshow(extrct_img)

    plt.legend(handles=legend_elements, loc='upper left', fontsize=20)


def labelme2xya(labelme_json, save_file, plot=False):
    
    def get_end_point(x, y, angle, length=30):# find the end point
        endy = y+length * math.sin(math.radians(angle))
        endx = x+length * math.cos(math.radians(angle))
        return tuple([int(endx), int(endy)])

    with open(labelme_json, 'r') as f:
        data = json.load(f)

    tissues = []
    magnets = []
    for polygon in data["shapes"]:  
        if polygon["label"].startswith("tissue") or polygon["label"] == "b1":
            tissues.append(np.asarray(polygon["points"]).flatten())           

        if polygon["label"].startswith("magnet") or polygon["label"] == "m1":
            magnets.append(np.asarray(polygon["points"]).flatten())

    seg_coord_tissues = pd.DataFrame(tissues)
    seg_coord_tissues["centroid"] = seg_coord_tissues.apply(lambda row: getCentroid(np.array(row).reshape(-1,2)), axis=1)

    seg_coord_mag = pd.DataFrame(magnets)
    seg_coord_mag["centroid"] = seg_coord_mag.apply(lambda row: getCentroid(np.array(row).reshape(-1,2)), axis=1)

    df = pd.DataFrame(data={"magnet_centroid":seg_coord_mag["centroid"], "tissue_centroid":seg_coord_tissues["centroid"] }, columns=["magnet_centroid", "tissue_centroid"])
    df["angle"] = df.apply(lambda row: math.degrees(math.atan2(row["magnet_centroid"][1]-row["tissue_centroid"][1], row["magnet_centroid"][0]-row["tissue_centroid"][0])), axis=1)
    df["x"] = df.apply(lambda row: row["tissue_centroid"][0], axis=1)
    df["y"] = df.apply(lambda row: row["tissue_centroid"][1], axis=1)
    
    df_xya = df[["x", "y", "angle"]]
    df_xya.to_csv(save_file, index=False)
    
    # plot with arrows
    if plot:
        thickness = 5
        tissue_color = "#ff6666"
        arrow_color = "#fdfd96"

        # Load image
        path_img = os.path.join(WAFER_DATA, data["imagePath"])
        if os.path.isfile(path_img):
            extrct_img = cv.imread(path_img)
        else:
            print (f"The file {path_img} does not exist.")

        plt.figure(figsize=(40, 40))
        for index, row in df_xya.iterrows():
            extrct_img = cv.circle(extrct_img, tuple([int(row["x"]), int(row["y"])]), 8, compute_rgb(tissue_color), -1)
            extrct_img = cv.arrowedLine(extrct_img, tuple([int(row["x"]), int(row["y"])]), get_end_point(row["x"], row["y"], row["angle"]), compute_rgb(arrow_color), thickness)

        plt.imshow(extrct_img)
    

def resize_input(labelme_init, im, im_fluo, ratio):
    print(f"Ratio: {ratio}")
    im = cv.resize(im, None, fx=ratio, fy=ratio)
    im_fluo = cv.resize(im_fluo, None, fx=ratio, fy=ratio)

    sections = list(zip(*(iter(labelme_init['shapes'][:-1]),) * 3))

    for i, [t, m, e] in enumerate(sections):
        t['points'] = (np.array(t['points']) * ratio).astype(int).tolist()
        m['points'] = (np.array(m['points']) * ratio).astype(int).tolist()
        e['points'] = (np.array(e['points']) * ratio).astype(int).tolist()

    labelme_init['shapes'][-1]["points"] = (np.array(labelme_init['shapes'][-1]["points"]) * ratio).astype(int).tolist()

    return labelme_init, im, im_fluo

