import configparser

def create_empty_magc():
    magc = {
        'sections': {},
        'rois': {},
        'magnets': {},
        'focus': {},
        'landmarksEM': {
            'source': {},
            'target': {}
            },
        }
    return magc

def read_magc(magc_path):
    config = configparser.ConfigParser()

    magc = create_empty_magc()

    with open(magc_path, 'r') as configfile:
        config.read_file(configfile)

    for header in config.sections():
        if header.startswith('sections.'):
            section_id = int(header.split('.')[1])
            magc['sections'][section_id] = {}
            for key,val in config.items(header):
                if key == 'polygon':
                    vals = [float(x) for x in val.split(',')]
                    poly_points = [[x,y] for x,y in zip(vals[::2], vals[1::2])]
                    magc['sections'][section_id]['polygon'] = poly_points
                elif key == 'center':
                    magc['sections'][section_id]['center'] = [float(x) for x in val.split(',')]
                elif key in ['area', 'compression']:
                    magc['sections'][section_id][str(key)] = float(val)
                elif key == 'angle':
                    magc['sections'][section_id][str(key)] = ((float(val)+90)%360) - 180

        elif header.startswith('rois.'):
            roi_id = int(header.split('.')[1])
            magc['rois'][roi_id] = {}
            for key,val in config.items(header):
                if key=='template':
                    magc['rois'][roi_id]['template'] = int(val)
                elif key == 'polygon':
                    vals = [float(x) for x in val.split(',')]
                    poly_points = [[x,y] for x,y in zip(vals[::2], vals[1::2])]
                    magc['rois'][roi_id]['polygon'] = poly_points
                elif key == 'center':
                    magc['rois'][roi_id]['center'] = [float(x) for x in val.split(',')]
                elif key in ['area']:
                    magc['rois'][roi_id][str(key)] = float(val)
                elif key == 'angle':
                    magc['rois'][roi_id][str(key)] = ((float(val)+90)%360) - 180

        elif header.startswith('magnets.'):
            magnet_id = int(header.split('.')[1])
            magc['magnets'][magnet_id] = {}
            for key,val in config.items(header):
                if key=='template':
                    magc['magnets'][magnet_id]['template'] = int(val)
                elif key=='location':
                    magc['magnets'][magnet_id]['location'] = [float(x) for x in val.split(',')]

        elif header.startswith('focus.'):
            focus_id = int(header.split('.')[1])
            magc['focus'][focus_id] = {}
            for key,val in config.items(header):
                if key=='template':
                    magc['focus'][focus_id]['template'] = int(val)
                elif key in ['location', 'polygon']:
                    vals = [float(x) for x in val.split(',')]
                    focus_points = [
                        [x,y]
                        for x,y in zip(vals[::2], vals[1::2])]
                    magc['focus'][focus_id]['polygon'] = focus_points

        elif header.startswith('landmarksEM.'):
            landmark_id = int(header.split('.')[1])
            magc['landmarksEM']['source'][landmark_id] = [float(x) for x in config.get(header, 'location').split(',')]

        elif header == 'serialorder':
            value = config.get('serialorder', 'serialorder')
            if value!='[]':
                magc['serialorder'] = [int(x) for x in value.split(',')]

        elif header == 'tsporder':
            value = config.get('tsporder', 'tsporder')
            if value!='[]':
                magc['tsporder'] = [int(x) for x in value.split(',')]

    return magc

'''
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
        1
            'template': x
            'polygon': [[x1,y1],...,[xn,yn]]
            'center': [x,y]
            'area': x (pixel)
        .   'angle': x (degree)
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
'''

magc_path = xxx
magc = read_magc(magc_path)
print(
    'magc dictionary',
    magc)