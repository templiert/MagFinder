import configparser


def create_empty_magc():
    magc = {
        "sections": {},
        "rois": {},
        "magnets": {},
        "focus": {},
        "landmarksEM": {"source": {}, "target": {}},
    }
    return magc


def read_magc(magc_path):
    config = configparser.ConfigParser()

    magc = create_empty_magc()

    with open(magc_path, "r") as configfile:
        config.read_file(configfile)

    for header in config.sections():
        if header.startswith("section."):
            section_dict = {}
            for key, val in config.items(header):
                if key == "polygon":
                    vals = [float(x) for x in val.split(",")]
                    poly_points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                    section_dict["polygon"] = poly_points
                elif key == "center":
                    section_dict["center"] = [float(x) for x in val.split(",")]
                elif key in ["area", "compression"]:
                    section_dict[str(key)] = float(val)
                elif key == "angle":
                    section_dict[str(key)] = ((float(val) + 90) % 360) - 180
            magc["sections"][int(header.split(".")[1])] = section_dict

        elif header.startswith("roi."):
            section_id = int(header.split(".")[1])
            roi_id = int(header.split(".")[2])
            if section_id not in magc["rois"]:
                magc["rois"][section_id] = {}
            roi_dict = {}
            for key, val in config.items(header):
                if key == "template":
                    roi_dict["template"] = int(val)
                elif key == "polygon":
                    vals = [float(x) for x in val.split(",")]
                    poly_points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                    roi_dict["polygon"] = poly_points
                elif key == "center":
                    roi_dict["center"] = [float(x) for x in val.split(",")]
                elif key in ["area"]:
                    roi_dict[str(key)] = float(val)
                elif key == "angle":
                    roi_dict[str(key)] = ((float(val) + 90) % 360) - 180
            magc["rois"][section_id][roi_id] = roi_dict

        elif header.startswith("magnet."):
            magnet_dict = {}
            for key, val in config.items(header):
                if key == "template":
                    magnet_dict["template"] = int(val)
                elif key == "location":
                    magnet_dict["location"] = [float(x) for x in val.split(",")]
            magc["magnets"][int(header.split(".")[1])] = magnet_dict

        elif header.startswith("focus."):
            focus_dict = {}
            for key, val in config.items(header):
                if key == "template":
                    focus_dict["template"] = int(val)
                elif key in ["location", "polygon"]:
                    vals = [float(x) for x in val.split(",")]
                    focus_points = [[x, y] for x, y in zip(vals[::2], vals[1::2])]
                    focus_dict["polygon"] = focus_points
            magc["focus"][int(header.split(".")[1])] = focus_dict

        elif header.startswith("landmarksEM."):
            landmark_id = int(header.split(".")[1])
            magc["landmarksEM"]["source"][landmark_id] = [
                float(x) for x in config.get(header, "location").split(",")
            ]

        elif header == "serialorder":
            value = config.get("serialorder", "serialorder")
            if value != "[]":
                magc["serialorder"] = [int(x) for x in value.split(",")]

        elif header == "tsporder":
            value = config.get("tsporder", "tsporder")
            if value != "[]":
                magc["tsporder"] = [int(x) for x in value.split(",")]

    return magc


"""
###############################
# format of the magc dictionary
###############################
magc
    'sections'
        1
            'polygon': [[x1,y1],...,[xn,yn]]
            'center': [x,y]
            'area': x (pixel)
            'compression': x
        .   'angle': x (degree)
        :
        n
    'rois'
        1 # section id
            1 # roi id (multiple rois per section)
                'template': x
                'polygon': [[x1,y1],...,[xn,yn]]
                'center': [x,y]
                'area': x (pixel)
            .   'angle': x (degree)
            :
        .   n
        :
        n
    'magnets'
        1
            'template': x
        .   'location': [x,y]
        :
        n
    'focus'
        1
            'template': x
        .   'polygon': [[x1,y1],...,[xn,yn]]
        :
        n
    'landmarksEM'
        'source'
            1: [x,y]
            .
            :
            n: [x,y]
        'target'
            1: [x,y]
            .
            :
            n: [x,y]
'serialorder': [x1,...,xn]
'tsporder': [x1,...,xn]
"""

magc_path = xxx
magc = read_magc(magc_path)
print(f"{magc=}")
