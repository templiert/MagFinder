import sys, os, shutil, time, threading
import datetime, java, jarray, copy, ConfigParser

import ij
from ij import IJ, ImagePlus, ImageStack, WindowManager

from ij.io import DirectoryChooser
from ij.gui import Roi, PolygonRoi, PointRoi, GenericDialog, NewImage
from fiji.util.gui import GenericDialogPlus
from ij.plugin import MontageMaker, CanvasResizer, ImageCalculator
from ij.plugin.frame import RoiManager
from java.lang import Math, Runtime
from java.lang import Exception as java_exception
from java.lang.reflect import Array
from java.util import HashSet
from java.awt import Rectangle, Color, Button, Checkbox
from java.awt.geom import AffineTransform, Point2D
from java.awt.geom.Point2D import Float as pFloat
from java.awt.event import KeyEvent, KeyAdapter, MouseAdapter, ActionListener
from mpicbg.models import RigidModel2D, AffineModel2D, PointMatch, Point
from Jama import Matrix

from org.apache.commons.io import FileUtils
from java.io import File, FileInputStream
from java.net import URL
from java.util.zip import GZIPInputStream, ZipFile
from java.nio.file import Files, Paths

###############
# I/O functions
###############
def get_OK(text):
    gd = GenericDialog('User prompt')
    gd.addMessage(text)
    gd.hideCancelButton()
    gd.enableYesNoCancel()
    focus_on_ok(gd)
    gd.showDialog()
    return gd.wasOKed()

def create_empty_magc():
    magc = {
        'sections': {},
        'rois': {},
        'magnets': {},
        'focus': {},
        'landmarksEM': {},
        'serialorder': [],
        'tsporder': []
        }
    return magc

def read_ini():
    global iniPath
    iniPaths = [os.path.join(experimentFolder,fileName)
        for fileName in os.listdir(experimentFolder)
        if os.path.splitext(fileName)[1] == '.ini']

    if iniPaths == []: # if no .ini file (user is creating a wafer by hand)
        IJ.log('No .ini file found. Creating an empty one')
        iniPath = os.path.join(
            experimentFolder,
            'segmented_wafer.ini')
        magc = create_empty_magc()
    else:
        config = ConfigParser.ConfigParser()
        iniPath = iniPaths[0]
        magc = {}

        with open(iniPath, 'rb') as configfile:
            config.readfp(configfile)

        for header in config.sections():
            if header == 'sections':
                magc['sections'] = {}

            elif header == 'rois':
                magc['rois'] = {}

            elif header == 'magnets':
                magc['magnets'] = {}

            elif header == 'focus':
                magc['focus'] = {}

            elif header == 'landmarksEM':
                magc['landmarksEM'] = {}

            elif 'sections.' in header:
                section_id = int(header.split('.')[1])
                magc['sections'][section_id] = {}
                for key,val in config.items(header):
                    if key == 'polygon':
                        vals = map(float, val.split(','))
                        poly_points = [[x,y] for x,y in zip(vals[::2], vals[1::2])]
                        magc['sections'][section_id]['polygon'] = poly_points
                    elif key == 'center':
                        magc['sections'][section_id]['center'] = map(float, val.split(','))
                    elif key in ['area', 'compression']:
                        magc['sections'][section_id][str(key)] = float(val)
                    elif key == 'angle':
                        magc['sections'][section_id][str(key)] = ((float(val)+90)%360) - 180

            elif 'rois.' in header:
                roi_id = int(header.split('.')[1])
                magc['rois'][roi_id] = {}
                for key,val in config.items(header):
                    if key=='template':
                        magc['rois'][roi_id]['template'] = int(val)
                    elif key == 'polygon':
                        vals = map(float, val.split(','))
                        poly_points = [[x,y] for x,y in zip(vals[::2], vals[1::2])]
                        magc['rois'][roi_id]['polygon'] = poly_points
                    elif key == 'center':
                        magc['rois'][roi_id]['center'] = map(float, val.split(','))
                    elif key in ['area']:
                        magc['rois'][roi_id][str(key)] = float(val)
                    elif key == 'angle':
                        magc['rois'][roi_id][str(key)] = ((float(val)+90)%360) - 180

            elif 'magnets.' in header:
                magnet_id = int(header.split('.')[1])
                magc['magnets'][magnet_id] = {}
                for key,val in config.items(header):
                    if key=='template':
                        magc['magnets'][magnet_id]['template'] = int(val)
                    elif key=='location':
                        magc['magnets'][magnet_id]['location'] = map(float, val.split(','))

            elif 'focus.' in header:
                focus_id = int(header.split('.')[1])
                magc['focus'][focus_id] = {}
                for key,val in config.items(header):
                    if key=='template':
                        magc['focus'][focus_id]['template'] = int(val)
                    elif key in ['location', 'polygon']:
                        vals = map(float, val.split(','))
                        focus_points = [
                            [x,y]
                            for x,y in zip(vals[::2], vals[1::2])]
                        magc['focus'][focus_id]['polygon'] = focus_points

            elif 'landmarksEM.' in header:
                landmark_id = int(header.split('.')[1])
                magc['landmarksEM'][landmark_id] = map(float, config.get(header, 'location').split(','))

            elif header == 'serialorder':
                if config.get('serialorder', 'serialorder') != '[]':
                    magc['serialorder'] = map(int, config.get('serialorder', 'serialorder').split(','))
                else:
                    magc['serialorder'] = []

            elif header == 'tsporder':
                if config.get('tsporder', 'tsporder') != '[]':
                    magc['tsporder'] = map(int, config.get('tsporder', 'tsporder').split(','))
                else:
                    magc['tsporder'] = []

    IJ.log('File successfully read: ' + iniPath)
    # IJ.log('*** MAGC FILE ***' + str(magc))
    return magc

def points_to_flat_string(points):
    points_flat = []
    for point in points:
        points_flat.append(point[0])
        points_flat.append(point[1])
    points_string = ','.join(
        map(lambda x: str(round(x,3)),
            points_flat))
    return points_string

def point_to_flat_string(point):
    flat_string = ','.join(
        map(lambda x: str(round(x,3)),
        point))
    return flat_string

def write_ini(magc):
    config = ConfigParser.ConfigParser()

    if 'sections' in magc:
        config.add_section('sections')
        config.set('sections', 'number', str(len(magc['sections'])))
        section_ids = sorted(magc['sections'].keys())
        for section_id in section_ids:
            header = 'sections.' + str(section_id).zfill(4)
            config.add_section(header)
            # polygon
            config.set(header, 'polygon',
                points_to_flat_string(
                    magc['sections'][section_id]['polygon']))
            # center
            config.set(header, 'center',
                point_to_flat_string(
                    magc['sections'][section_id]['center']))
            # area
            config.set(header, 'area',
                str(magc['sections'][section_id]['area']))
            # angle
            config.set(header,'angle',
                str( (((magc['sections'][section_id]['angle'])-90)%360)-180))
            # compression
            config.set(header, 'compression',
                str(magc['sections'][section_id]['compression']))

    if 'rois' in magc:
        config.add_section('rois')
        config.set('rois', 'number', str(len(magc['rois'])))
        roi_ids = sorted(magc['rois'].keys())
        for roi_id in roi_ids:
            header = 'rois.' + str(roi_id).zfill(4)
            config.add_section(header)
            # template
            config.set(header, 'template',
                str(magc['rois'][roi_id]['template']))
            # polygon
            config.set(header, 'polygon',
                points_to_flat_string(
                    magc['rois'][roi_id]['polygon']))
            # center
            config.set(header, 'center',
                point_to_flat_string(
                    magc['rois'][roi_id]['center']))
            # area
            config.set(header, 'area',
                str(magc['rois'][roi_id]['area']))
            # angle
            config.set(header, 'angle',
                str( (((magc['rois'][roi_id]['angle'])-90)%360)-180))

    if 'focus' in magc:
        config.add_section('focus')
        config.set('focus', 'number', str(len(magc['focus'])))
        focus_ids = sorted(magc['focus'].keys())
        for focus_id in focus_ids:
            header = 'focus.' + str(focus_id).zfill(4)
            config.add_section(header)
            # template
            config.set(header, 'template',
                str(magc['focus'][focus_id]['template']))
            # polygon
            config.set(header, 'polygon',
                points_to_flat_string(
                    magc['focus'][focus_id]['polygon']))

    if 'magnets' in magc:
        config.add_section('magnets')
        config.set('magnets', 'number', str(len(magc['magnets'])))
        magnet_ids = sorted(magc['magnets'].keys())
        for magnet_id in magnet_ids:
            header = 'magnets.' + str(magnet_id).zfill(4)
            config.add_section(header)
            # template
            config.set(header, 'template',
                str(magc['magnets'][magnet_id]['template']))
            # location
            config.set(header, 'location',
                point_to_flat_string(
                    magc['magnets'][magnet_id]['location']))

    if 'landmarksEM' in magc:
        config.add_section('landmarksEM')
        config.set('landmarksEM', 'number', str(len(magc['landmarksEM'])))
        landmarksEM_ids = sorted(magc['landmarksEM'].keys())
        for landmarksEM_id in landmarksEM_ids:
            header = 'landmarksEM.' + str(landmarksEM_id).zfill(4)
            config.add_section(header)
            config.set(header, 'location',
                point_to_flat_string(
                    magc['landmarksEM'][landmarksEM_id]))

    if 'serialorder' in magc:
        config.add_section('serialorder')
        if magc['serialorder'] == []:
            config.set(
                'serialorder',
                'serialorder',
                '[]')
        else:
            config.set(
                'serialorder',
                'serialorder',
                ','.join(map(str,
                    magc['serialorder'])))

    if 'tsporder' in magc:
        config.add_section('tsporder')
        if magc['tsporder'] == []:
            config.set(
                'tsporder',
                'tsporder',
                '[]')
        else:
            config.set(
                'tsporder',
                'tsporder',
                ','.join(map(str,
                    magc['tsporder'])))

    with open(iniPath, 'wb') as configfile:
       config.write(configfile)

#######################
# Geometrical functions
#######################
def is_image_padded(im):
    im_ref_1 = im.duplicate()
    im_ref_2 = im.duplicate()

    w = im.getWidth()
    h = im.getHeight()

    padding_factor_check = PADDING_FACTOR - 0.01

    im_white = NewImage.createByteImage(
        'black_padding',
        int(w * (1-2*padding_factor_check)),
        int(h * (1-2*padding_factor_check)),
        1,
        NewImage.FILL_WHITE)

    IJ.run(
        im_white,
        'Canvas Size...',
        ('width=' + str(w)
            + ' height=' + str(h)
            + ' position=Center zero'))

    ic = ImageCalculator()
    ic.run(
        'AND',
        im_ref_2,
        im_white)

    ic.run(
        'Subtract',
        im_ref_1,
        im_ref_2)

    return (im_ref_1.getStatistics().max == 0)

def pad_image(im, padding_factor):
    w1 = im.getWidth()
    h1 = im.getHeight()

    w2 = w1/(1-2*padding_factor)
    h2 = h1/(1-2*padding_factor)

    IJ.run(
        im,
        'Canvas Size...',
        ('width=' + str(w2)
            + ' height=' + str(h2)
            + ' position=Center zero'))

def load_pad_save(im_path, padding_factor):
    im = IJ.openImage(im_path)
    if not is_image_padded(im):
        IJ.log('Image needs to be padded')
        pad_image(im, padding_factor)
        IJ.save(im, im_path)
        IJ.log('Image has been padded')
    else:
        IJ.log('Saved image is already padded')
    return im

def crop(im,roi):
    ip = im.getProcessor()
    ip.setRoi(roi)
    im = ImagePlus(
        im.getTitle() + '_Cropped',
        ip.crop())
    return im

def longest_diagonal(corners):
    maxDiag = 0
    for corner1 in corners:
        for corner2 in corners:
            maxDiag = max(
                get_distance(corner1, corner2),
                maxDiag)
    return int(maxDiag)

def get_distance(corner1, corner2):
    return Math.sqrt(
        (corner2[0]-corner1[0]) * (corner2[0]-corner1[0])
        + (corner2[1]-corner1[1]) * (corner2[1]-corner1[1]))

def polygonroi_from_points(points):
    xPoly = [point[0] for point in points]
    yPoly = [point[1] for point in points]
    return PolygonRoi(xPoly, yPoly, PolygonRoi.POLYGON)

def centroid(points):
    return polygonroi_from_points(points).getContourCentroid()

def polygon_area(points):
    return polygonroi_from_points(points).getStatistics().pixelCount

def get_model_from_points(sourcePoints, targetPoints):
    rigidModel = RigidModel2D()
    pointMatches = HashSet()
    for a in zip(sourcePoints, targetPoints):
        pm = PointMatch(
            Point([a[0][0], a[0][1]]),
            Point([a[1][0], a[1][1]]))
        pointMatches.add(pm)
    rigidModel.fit(pointMatches)
    return rigidModel

def get_angle(line):
    diff = [
        line[0] - line[2],
        line[1] - line[3]]
    theta = Math.atan2(diff[1], diff[0])
    return theta

def rotate(im, angleDegree):
    IJ.run(im, 'Rotate... ',
        'angle=' + str(angleDegree) + ' grid=1 interpolation=Bilinear')

def section_to_list(pointList): # [[1,2],[5,8]] to [1,2,5,8]
    l = jarray.array(
        2 * len(pointList) * [0],
        'd')
    for id, point in enumerate(pointList):
        l[2*id] = point[0]
        l[2*id+1] = point[1]
    return l

def list_to_section(l): # [1,2,5,8] to [[1,2],[5,8]]
    pointList = []
    for i in range(len(l)/2):
        pointList.append([l[2*i], l[2*i+1]])
    return pointList

def apply_transform(section, aff):
    sourceList = section_to_list(section)
    targetList = jarray.array(len(sourceList) * [0], 'd')
    aff.transform(sourceList, 0, targetList, 0, len(section))
    targetSection = list_to_section(targetList)
    return targetSection

##############################
### Affine transform utils ###

def affine_t(x_in, y_in, x_out, y_out):
    X = Matrix(jarray.array(
        [[x, y, 1] for (x,y) in zip(x_in, y_in)],
        java.lang.Class.forName('[D')))
    Y = Matrix(jarray.array(
        [[x, y, 1] for (x,y) in zip(x_out, y_out)],
        java.lang.Class.forName('[D')))
    aff = X.solve(Y)
    return aff

def apply_affine_t(x_in, y_in, aff):
    X = Matrix(jarray.array(
        [[x, y, 1] for (x,y) in zip(x_in, y_in)],
        java.lang.Class.forName('[D')))
    Y = X.times(aff)
    x_out = [float(y[0]) for y in Y.getArrayCopy()]
    y_out = [float(y[1]) for y in Y.getArrayCopy()]
    return x_out, y_out

def invert_affine_t(aff):
    return aff.inverse()

#############################
### Propagation functions ###

def points_to_xy(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return x,y

def propagate_points(source_section, source_points, target_section):
    source_section_x, source_section_y = points_to_xy(source_section)
    source_points_x, source_points_y = points_to_xy(source_points)
    target_section_x, target_section_y = points_to_xy(target_section)
    aff = affine_t(
        source_section_x, source_section_y,
        target_section_x, target_section_y)
    target_points_x, target_points_y = apply_affine_t(
        source_points_x,
        source_points_y,
        aff)
    target_points = [
        [x,y]
        for x,y in zip(target_points_x, target_points_y)]
    # IJ.log(
        # 'source_section - ' + str(source_section) + '\n'
        # + 'target_section - ' + str(target_section) + '\n'
        # + 'source_points - ' + str(source_points) + '\n'
        # + 'target_points - ' + str(target_points) + '\n')
    return target_points

def get_indexes_from_user_string(userString):
    '''inspired by the substackMaker of ImageJ \n
    https://imagej.nih.gov/ij/developer/api/ij/plugin/SubstackMaker.html
    Enter a range (2-30), a range with increment (2-30-2), or a list (2,5,3)
    '''
    userString = userString.replace(' ', '')
    if ',' in userString and '.' in userString:
        return None
    elif ',' in userString:
        splitIndexes = [int(splitIndex) for splitIndex in userString.split(',')
                        if splitIndex.isdigit()]
        if len(splitIndexes) > 0:
            return splitIndexes
    elif '-' in userString:
        splitIndexes = [int(splitIndex) for splitIndex in userString.split('-')
                        if splitIndex.isdigit()]
        if len(splitIndexes) == 2 or len(splitIndexes) == 3:
            splitIndexes[1] = splitIndexes[1] + 1 # inclusive is more natural (2-5 = 2,3,4,5)
            return range(*splitIndexes)
    elif userString.isdigit():
        return [int(userString)]
    return None

################
# TSP Operations
################
def download_unzip(url, target_path):
    # download, unzip, clean
    IJ.log('Downloading TSP solver from ' + str(url))

    gz_path = os.path.join(
        pluginFolder,
        'temp.gz')
    try:
        f = File(gz_path)
        FileUtils.copyURLToFile(
            URL(url),
            f)
        gis = GZIPInputStream(
                FileInputStream(
                    gz_path))
        Files.copy(
            gis,
            Paths.get(target_path))
        gis.close()
        os.remove(gz_path)
    except (Exception, java_exception) as e:
        IJ.log(
            'Failed to download from ' + str(url)
            + ' due to ' + str(e))

def order_from_mat(mat, rootFolder, solverPath, solutionName = ''):
    tsplibPath = os.path.join(rootFolder, 'TSPMat.tsp')
    save_mat_to_TSPLIB(mat, tsplibPath)
    solutionPath = os.path.join(
        rootFolder,
        'solution_' + solutionName + '.txt')
    if os.path.isfile(solutionPath):
        os.remove(solutionPath)
    # adding " is needed if there are spaces in the paths
    command = '"' + solverPath + '" -o "' + solutionPath + '" "' + tsplibPath + '"'
    # IJ.log('TSP solving command ' + str(command))

    process = Runtime.getRuntime().exec(command)

    while not os.path.isfile(solutionPath):
        time.sleep(1)
        IJ.log(
            'Computing TSP solution with the '
            + os.path.basename(solverPath).replace('.exe','')
            + ' solver ...')
    time.sleep(1)

    order = []
    if 'linkern.exe' in solverPath:
        with open(solutionPath, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            order.append(int(line.split(' ')[0]))

    elif 'concorde.exe' in solverPath:
        with open(solutionPath, 'r') as f:
            result_line = f.readlines()[1]
            result_line = result_line.replace('\n','')
            result_line_splitted = result_line.split(' ')
            result_line_splitted.remove('')
            order = map(
                int,
                result_line_splitted)

    # remove the dummy city 0 and apply a -1 offset
    order.remove(0)
    for id, o in enumerate(order):
        order[id] = o-1

    costs = []
    for id, o in enumerate(order[:-1]):
        o1, o2 = sorted([order[id], order[id+1]]) # sorting because [8, 6] is not in the matrix, but [6,8] is
        cost = mat[o1][o2]
        # IJ.log( 'order cost ' + str(order[id]) + '_' +  str(order[id+1]) + '_' + str(cost))
        costs.append(cost)
    totalCost = sum(costs)
    IJ.log(
        'The total stage travel of the optimal order is '
        + str(int(totalCost)) + ' pixels.')
    IJ.log(
        'The total stage travel of the non-optimized order is '
        + str(int(sum(
            [mat[t][t+1]
                for t in range(len(order) - 1)] )))
        + ' pixels.')

    # delete temporary files
    if os.path.isfile(solutionPath):
        os.remove(solutionPath)
    if os.path.isfile(tsplibPath):
        os.remove(tsplibPath)

    return order, costs

def save_mat_to_TSPLIB(mat, path):
    # the matrix is a distance matrix
    # IJ.log('Entering save_mat_to_TSPLIB')
    n = len(mat)
    f = open(path, 'w')
    f.write('NAME: Section_Similarity_Data' + '\n')
    f.write('TYPE: TSP' + '\n')
    f.write('DIMENSION: ' + str(n + 1) + '\n')
    f.write('EDGE_WEIGHT_TYPE: EXPLICIT' + '\n')
    f.write('EDGE_WEIGHT_FORMAT: UPPER_ROW' + '\n')
    f.write('NODE_COORD_TYPE: NO_COORDS' + '\n')
    f.write('DISPLAY_DATA_TYPE: NO_DISPLAY' + '\n')
    f.write('EDGE_WEIGHT_SECTION' + '\n')

    distances = [0]*n #dummy city
    for i in range(n):
        for j in range(i+1, n, 1):
            distance = mat[i][j]
            distances.append(int(float(distance)))

    for id, distance in enumerate(distances):
        f.write(str(distance))
        if (id + 1)%10 == 0:
            f.write('\n')
        else:
            f.write(' ')
    f.write('EOF' + '\n')
    f.close()

def init_mat(n, initValue = 0):
    a = Array.newInstance(java.lang.Float,[n, n])
    for i in range(n):
        for j in range(n):
            a[i][j] = initValue
    return a

def handleKeypressO():
    section_keys = magc['sections'].keys()
    n_points = len(section_keys)
    if n_points<2:
        IJ.showMessage(
            'Warning',
            'You just pressed [o] to compute the order that minimizes stage travel.'
            +'\n\nThere is no stage travel to be optimized because there are less than 2 sections.')
        return
    elif n_points<8:
        solverPath = concordePath
    else:
        solverPath = linkernPath

    if not (solverPath and cygwindllPath):
        IJ.showMessage(
            'Error',
            ('The TSP solver or cygwin1.dll were not found in the Fiji plugins directory when Fiji started.'
            + '\n\nCannot compute the optimal path to minimize microscope stage movement.'))
        return
    compute_save_tsp_order()

def compute_save_tsp_order():
    section_keys = magc['sections'].keys()
    n_points = len(section_keys)

    if n_points<2:
        return
    if n_points<8:
        solverPath = concordePath
    else:
        solverPath = linkernPath

    if not solverPath:
        IJ.log('Could not compute the stage-movement-minimizing order because the solver or cygwin1.dll are missing')
        return

    try:
        center_points = [pFloat(*magc['sections'][section_key]['center'])
            for section_key in section_keys]

        # initialize distance matrix
        distances = init_mat(
            n_points,
            initValue=999999)

        # fill distance matrix
        for a in range(n_points):
            for b in range(0,a+1,1):
                distances[b][a] = center_points[a].distance(center_points[b])

        order,_ = order_from_mat(
            distances,
            experimentFolder,
            solverPath)
        magc['tsporder'] = order
        IJ.log('The optimal section order to minimize stage travel is: ' + str(order))
    except (Exception, java_exception) as e:
        IJ.log('The path to minimize stage movement could not be computed: ' + str(e))
        pass

###############
# GUI Functions
###############

class ListenToKey(KeyAdapter):
    def keyPressed(this, event):
        event.consume()
        if globalMode:
            handleKeypressGlobalMode(event)
        else:
            handleKeypressLocalMode(event)

class ListenToMouseWheel(MouseAdapter):
    def mouseWheelMoved(this, mouseWheelEvent):
        if globalMode:
            handleMouseWheelGlobalMode(mouseWheelEvent)
        else:
            handleMouseWheelLocalMode(mouseWheelEvent)

class ButtonClick(ActionListener):
    def actionPerformed(self, event):
        source = event.getSource()
        stringField = source.getParent().getStringFields()[0]
        stringField.setText(source.label.split(' ')[-1])
        # # # if 'All sections' in source.label:
            # # # # stringField.setText('0-' + str(get_section_number()-1))
        # # # elif 'First half of the sections' in source.label:
            # # # stringField.setText('0-' + str(int((get_section_number()-1)/2)))
        # # # elif 'Every second section' in source.label:
            # # # stringField.setText('0-' + str(get_section_number()-1) + '-' + str(2))

def handleMouseWheelLocalMode(mouseWheelEvent):
    mouseWheelEvent.consume()
    if mouseWheelEvent.isControlDown():
        move_roi_manager_selection(10 * mouseWheelEvent.getWheelRotation())
    else:
        move_roi_manager_selection(mouseWheelEvent.getWheelRotation())

def handleMouseWheelGlobalMode(mouseWheelEvent):
    mouseWheelEvent.consume()
    if mouseWheelEvent.isShiftDown():
        if mouseWheelEvent.getWheelRotation() == 1:
            move_fov('right')
        else:
            move_fov('left')
    elif (not mouseWheelEvent.isShiftDown()) and (not mouseWheelEvent.isControlDown()):
        if mouseWheelEvent.getWheelRotation() == 1:
            move_fov('up')
        else:
            move_fov('down')
    elif mouseWheelEvent.isControlDown():
        if mouseWheelEvent.getWheelRotation() == 1:
            IJ.run('Out [-]')
        elif mouseWheelEvent.getWheelRotation() == -1:
            IJ.run('In [+]')

def move_fov(a):
    im = IJ.getImage()
    canvas = im.getCanvas()
    r = canvas.getSrcRect()
    # adjust increment depending on zoom level
    increment = int(40/float(canvas.getMagnification()))
    xPixelIncrement = 0
    yPixelIncrement = 0
    if a=='right':
        xPixelIncrement = increment
    elif a== 'left':
        xPixelIncrement = -increment
    elif a== 'up':
        yPixelIncrement = increment
    elif a== 'down':
        yPixelIncrement = -increment
    newR = Rectangle(
        min(max(0, r.x + xPixelIncrement), im.getWidth()-r.width),
        min(max(0, r.y + yPixelIncrement), im.getHeight()-r.height),
        r.width,
        r.height)
    canvas.setSourceRect(newR)
    im.updateAndDraw()

def get_closest_sections(x,y):
    sectionCenters = [
        magc['sections'][key]['center']
        for key in sorted(magc['sections'].keys())]
    distances = sorted(
        [[get_distance([x,y], sectionCenter), id]
        for id,sectionCenter in enumerate(sectionCenters)])
    return [d[1] for d in distances]

def handleKeyPressGlobalModeA():
    global magc
    manager = get_roi_manager()
    waferIm = IJ.getImage()
    drawnRoi = waferIm.getRoi()
    if not drawnRoi:
        IJ.log('Please draw something before pressing key [a] for adding')
    else:
        poly = drawnRoi.getFloatPolygon()
        points = [[x,y] for x,y in zip(poly.xpoints, poly.ypoints)]
        nameSuggestions = []
        managerRois = manager.getRoisAsArray()
        if len(points) == 2:
            IJ.log('Adding a landmark or magnet center')

            # suggest the 2 nearest sections if existing, plus the n+1 magnet if no section?
            # is magnet without section possible? let's decide that no
            closestSections = get_closest_sections(*points[0])
            suggestedMagnetNames = ['magnet-' + str(i).zfill(4) for i in closestSections[:3]]
            nameSuggestions = nameSuggestions + suggestedMagnetNames

            landmarksEM = [
                managerRoi
                for managerRoi in managerRois
                if 'landmark_em' in managerRoi.getName()]
            nameSuggestions.append(
                'landmark_em-' + str(len(landmarksEM)).zfill(4))

        else:
            polysContainingDrawnRoi = []
            # checking if drawnRoi is contained in another roi
            # in which case then it is a ROI inside a section
            for managerRoi in managerRois:
                managerRoiPoly = managerRoi.getFloatPolygon()
                if managerRoiPoly.contains(*points[0]):
                    if not (False in [managerRoiPoly.contains(*point) for point in points]):
                        polysContainingDrawnRoi.append(managerRoi)

            if polysContainingDrawnRoi:
                for polyContainingDrawnRoi in polysContainingDrawnRoi:
                    polyClass, polyKey = polyContainingDrawnRoi.getName().split('-')[:2]
                    if polyClass == 'section':
                        nameSuggestions = ['roi-' + str(polyKey).zfill(4)]
            else:
                #######################################
                # check the indices of current sections
                sectionKeys = magc['sections'].keys()
                if len(sectionKeys) > 0:
                    sectionMaxKey = max(sectionKeys)

                    # suggest next max index section but also all intermediate
                    # section indices that might have been deleted
                    for i in range(sectionMaxKey):
                        roiNames = [roi.getName() for roi in managerRois]
                        roiName = 'section-' + str(i).zfill(4)
                        if not (roiName in roiNames):
                            nameSuggestions.append(roiName)

                    sectionName = 'section-' + str(sectionMaxKey+1).zfill(4)
                    nameSuggestions.append(sectionName)
                else:
                    nameSuggestions.append('section-0000')

                # roiKeys = magc['rois'].keys()
                # if len(roiKeys) > 0:
                    # roiMaxKey = max(roiKeys)
                    # roiName = 'roi-' + str(roiMaxKey+1).zfill(4)
                    # nameSuggestions.append(roiName)
                # else:
                    # nameSuggestions.append('roi-0000')

                # sectionSet = set(magc['sections'].keys())
                # roiSet = set(magc['rois'].keys())

                # if len(tissueSet - magnetSet) > 0:
                    # IJ.log('Missing Magnet parts ' + str(tissueSet - magnetSet))
                # if len(magnetSet - tissueSet) > 0:
                    # IJ.log('Missing Tissue parts ' + str(magnetSet - tissueSet))

                # for missingMagnet in tissueSet - magnetSet:
                    # nameSuggestions.append('magnet-' + str(missingMagnet).zfill(4))
                # for missingTissue in magnetSet - tissueSet:
                    # nameSuggestions.append('tissue-' + str(missingMagnet).zfill(4))

                nameSuggestions = sorted(set(nameSuggestions)) # to remove duplicates
                #######################################

        gd = GenericDialog('Validate the name of the annotation')
        gd.addRadioButtonGroup(
            'Name of the annotation',
            nameSuggestions,
            len(nameSuggestions),
            1,
            nameSuggestions[0])

        focus_on_ok(gd)
        gd.showDialog()
        if gd.wasCanceled():
            return
        checkbox = gd.getRadioButtonGroups()[0].getSelectedCheckbox()
        if checkbox:
            roiName = checkbox.label
            drawnRoi.setName(roiName)
            roiId = int(roiName.split('-')[1])

            if 'section-' in roiName:
                magc['sections'][roiId] = fill_section_dict(drawnRoi)
                waferIm.killRoi()
                drawnRoi = PolygonRoi(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    Roi.POLYGON)
                drawnRoi.setName(roiName)
                waferIm.setRoi(drawnRoi)
                drawnRoi.setStrokeColor(Color.blue)

            elif 'roi-' in roiName:
                magc['rois'][roiId] = fill_roi_dict(drawnRoi)
                waferIm.killRoi()
                drawnRoi = PolygonRoi(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    Roi.POLYGON)
                drawnRoi.setName(roiName)
                waferIm.setRoi(drawnRoi)
                drawnRoi.setStrokeColor(Color.yellow)

            elif 'magnet' in roiName:
                magc['magnets'][roiId] = {
                    'template': -1,
                    'location': points[0]
                    }
                drawnRoi = PointRoi(*points[0])
                drawnRoi.setName(roiName)
                waferIm.killRoi()
                waferIm.setRoi(drawnRoi)

            elif 'landmark_em' in roiName:
                magc['landmarksEM'][roiId] = points[0]
                drawnRoi = PointRoi(*points[0])
                drawnRoi.setName(roiName)
                waferIm.killRoi()
                waferIm.setRoi(drawnRoi)
                # manager.addRoi(drawnRoi)

            if 'magnet' in roiName: #magnets can be overwritten, not the others
                overwrite_rois([drawnRoi])
            else:
                drawnRoi.setHandleSize(SIZE_HANDLE)
                manager.addRoi(drawnRoi)
            # reorder manager
            reorder_roi_manager()
            # update manager slider
            roi_manager_scroll_bottom() # there is a problem ...
            waferIm.killRoi()
            # manager.runCommand('Deselect')
            IJ.log('Annotation ' + roiName + ' added')

            # update magc
            manager_to_magc_global()
        else:
            IJ.log('No selection made: the section was not added')

def handleKeypressGlobalMode(keyEvent):
    global magc
    manager = get_roi_manager()
    keycode = keyEvent.getKeyCode()

    if keycode == KeyEvent.VK_S:
        save_all_global_rois()

    if keycode == KeyEvent.VK_G:
        toggle_fill(label='section')

    if keycode == KeyEvent.VK_B:
        toggle_fill(label='roi')

    if keycode == KeyEvent.VK_M:
        handleKeypressGlobalModeM()

    if keycode == KeyEvent.VK_O:
        save_all_global_rois()
        handleKeypressO()
        save_all_global_rois()

    if keycode == KeyEvent.VK_H:
        IJ.showMessage(
            'Help for global mode',
            helpMessageGlobalMode)

    if keycode == KeyEvent.VK_Q:
        manager_to_magc_global()
        compute_save_tsp_order()
        close_global_mode()

    if keycode == KeyEvent.VK_T:
        close_global_mode()
        IJ.log('Entering local mode ...')
        start_local_mode()

    if keycode == KeyEvent.VK_A:
        handleKeyPressGlobalModeA()

    if keycode == KeyEvent.VK_EQUALS or keycode == KeyEvent.VK_UP:
        IJ.run('In [+]')
    if keycode == KeyEvent.VK_MINUS or keycode == KeyEvent.VK_DOWN:
        IJ.run('Out [-]')

def get_section_number():
    manager = get_roi_manager()
    return len(
        [1 for roi in manager.getRoisAsArray()
            if 'section' in roi.getName()])

def get_annotation_type(annotationName):
    for annotationType in ANNOTATION_TYPES:
        if annotationType in annotationName:
            return annotationType
    return None

def reorder_roi_manager():
    manager = get_roi_manager()
    allRois = manager.getRoisAsArray()
    if len(allRois) == 0:
        return

    roiNames = [roi.getName() for roi in allRois]
    manager.reset()
    # add sections, rois and magnets
    allRoiNameTrailingNumbers = [
        int(roi.getName().split('-')[1])
        for roi in allRois
        if not ('landmark' in roi.getName())]
    ids = sorted(set(allRoiNameTrailingNumbers))
    for id in ids:
        manager.addRoi(
            [roi for roi in allRois
                if roi.getName() == 'section-' + str(id).zfill(4)][0]) # it assumes that there cannot be a magnet without a section

        theRoiName = 'roi-' + str(id).zfill(4)
        if theRoiName in roiNames:
            index = roiNames.index(theRoiName)
            manager.addRoi(allRois[index])

        theMagnetName = 'magnet-' + str(id).zfill(4)
        if theMagnetName in roiNames:
            index = roiNames.index(theMagnetName)
            manager.addRoi(allRois[index])

        theFocusName = 'focus-' + str(id).zfill(4)
        if theFocusName in roiNames:
            index = roiNames.index(theFocusName)
            manager.addRoi(allRois[index])

    # add landmarks
    for id,roi in sorted(enumerate(allRois)):
        if 'landmark' in roi.getName():
            manager.addRoi(roi)

def overwrite_rois(newRois):
    # if there is already a roi with the same name
    # then we reset the roiManager, restore all rois
    # except the one to add, and we add ours.
    # Dirty workaround because select/runCommand('Delete')
    # does not work, probably a thread timing issue
    manager = get_roi_manager()
    # allRois = copy.deepcopy(manager.getRoisAsArray())
    allRois = manager.getRoisAsArray()
    manager.reset()
    newRoisNames = [roi.getName() for roi in newRois]
    finalRois = (
        [roi for roi in allRois
            if not (roi.getName() in newRoisNames)]
        + newRois)
    for roi in finalRois:
        manager.addRoi(roi)

def delete_roi(roiName):
    manager = get_roi_manager()
    allRois = manager.getRoisAsArray()
    manager.reset()
    for roi in allRois:
        if roi.getName() != roiName:
            manager.addRoi(roi)

def reassign_roi_slices_after_section_deletion(i):
    manager = get_roi_manager()
    allRois = manager.getRoisAsArray()
    manager.reset()
    for roi in allRois:
        roi_position_in_stack = roi.getPosition()
        if roi_position_in_stack < i+1:
            roi.setPosition(roi_position_in_stack)
        else:
            roi.setPosition(roi_position_in_stack-1)
        manager.addRoi(roi)

def get_roi_index_from_current_slice():
    waferIm = IJ.getImage()
    sliceLabel = (waferIm
        .getImageStack()
        .getSliceLabel(
            waferIm.getSlice()))
    sectionId = int(sliceLabel.split('-')[-1])
    manager = get_roi_manager()

    # try to find a roi associated with the current slice
    id_s = get_roi_index_by_name('section-' + str(sectionId).zfill(4))
    if id_s is not None:
        return id_s

    id_r = get_roi_index_by_name('roi-' + str(sectionId).zfill(4))
    if id_r is not None:
        return id_r

    id_m = get_roi_index_by_name('magnet-' + str(sectionId).zfill(4))
    if id_m is not None:
        return id_m

    id_f = get_roi_index_by_name('focus-' + str(sectionId).zfill(4))
    if id_f is not None:
        return id_f

    # if no roi associated to the slice, select roi 0 if existing
    if manager.getCount()>0:
        return 0

    return None

def move_roi_manager_selection(n):
    manager = get_roi_manager()
    if manager.getCount() == 0:
        return
    selectedIndex = manager.getSelectedIndex()
    if selectedIndex == -1:
        roiId = get_roi_index_from_current_slice()
        if roiId is not None:
            set_roi_and_update_roi_manager(roiId)
    else:
        manager.runCommand('Update')
        if n < 0:
            set_roi_and_update_roi_manager(max(0, selectedIndex + n))
        elif n > 0:
            set_roi_and_update_roi_manager(min(manager.getCount()-1, selectedIndex + n))

def select_roi_by_name(roiName):
    manager = get_roi_manager()
    roiIndex = [roi.getName()
        for roi in manager.getRoisAsArray()].index(roiName)
    manager.select(roiIndex)

def handleKeypressGlobalModeM():
    waferIm = IJ.getImage()
    flattened = waferIm.flatten()
    flattened_path = os.path.join(
        experimentFolder,
        'overview_global.jpg')
    IJ.save(
        flattened,
        flattened_path)
    IJ.log(
        'Flattened global image saved to '
        + str(flattened_path))

def handleKeypressLocalModeM():
    # save an overview montage of sections
    # and rois, focus

    global magc
    manager = get_roi_manager()
    managerRois = manager.getRoisAsArray()
    waferIm = IJ.getImage()
    # ---
    ims = waferIm.getStack()
    flattened_ims = []
    montageMaker = MontageMaker()

    n_slices = waferIm.getNSlices()

    n_rows = int(n_slices**0.5)
    n_cols = n_slices//n_rows
    if n_rows*n_cols < n_slices:
        n_rows+=1

    #adjust handle/stroke size depending on image dimensions
    im_w = waferIm.getWidth()
    montage_factor = 1 if im_w<LOCAL_SIZE_STANDARD else LOCAL_SIZE_STANDARD/float(im_w)

    for i in range(n_slices):
        im_p = ims.getProcessor(i+1).duplicate()
        id_slice = int(ims.getSliceLabel(i+1).split('-')[-1])
        flattened = ImagePlus('flattened', im_p)

        for roi in managerRois:
            if int(roi.getName().split('-')[1]) == id_slice:
                cloned_roi = roi.clone()

                if im_w <LOCAL_SIZE_STANDARD:
                    handle_size = 5
                    stroke_size = 3
                else:
                    handle_size = 5*int(im_w/LOCAL_SIZE_STANDARD)
                    stroke_size = 3*im_w/LOCAL_SIZE_STANDARD

                cloned_roi.setHandleSize(handle_size)
                cloned_roi.setStrokeWidth(stroke_size)
                flattened.setRoi(cloned_roi)
                flattened = flattened.flatten()

        flattened.setTitle('section-' + str(id_slice).zfill(4))
        flattened_ims.append(flattened)

    flattened_stack = ImageStack(
        flattened_ims[0].getWidth(),
        flattened_ims[0].getHeight())
    for flattened in flattened_ims:
        flattened_stack.addSlice(
            flattened.getTitle(),
            flattened.getProcessor())
    montage_stack = ImagePlus(
        'Montage',
        flattened_stack)

    montage = montageMaker.makeMontage2(
        montage_stack,
        n_rows, n_cols,
        montage_factor,
        1, montage_stack.getNSlices(), 1,
        3, True)
    flattened_path = os.path.join(
        experimentFolder,
        'overview_local.jpg')
    IJ.save(
        montage,
        flattened_path)
    del flattened_ims
    IJ.log(
        'Flattened local image saved to '
        + str(flattened_path))

def handleKeypressLocalModeA():
    global magc
    manager = get_roi_manager()
    waferIm = IJ.getImage()
    drawnRoi = waferIm.getRoi()
    if not drawnRoi:
        IJ.showMessage(
            'Info',
            ('Please draw something before pressing [a].'
            + '\nAfter closing this message you can press [h] for help.'))
        return

    poly = drawnRoi.getFloatPolygon()
    points = [[x,y] for x,y in zip(poly.xpoints, poly.ypoints)]
    managerRois = manager.getRoisAsArray()

    sliceLabel = (waferIm
        .getImageStack()
        .getSliceLabel(
            waferIm.getSlice()))
    currentSection = int(sliceLabel.split('-')[1])
    # alternatively, this could have been found from which
    # roi is set to this slice (might be faulty with weird cases)

    nameSuggestions = []
    if len(points) < 3:
        nameSuggestions.append('magnet-' + str(currentSection).zfill(4))
    else:
        nameSuggestions.append('section-' + str(currentSection).zfill(4))
        nameSuggestions.append('roi-' + str(currentSection).zfill(4))
    nameSuggestions.append('focus-' + str(currentSection).zfill(4))

    gd = GenericDialog('Validate the name of the annotation')
    gd.addRadioButtonGroup(
        'Name of the annotation',
        nameSuggestions,
        len(nameSuggestions),
        1,
        nameSuggestions[1] if len(nameSuggestions)>1 else nameSuggestions[0])
    focus_on_ok(gd)
    gd.showDialog()

    if gd.wasCanceled():
        return

    checkbox = gd.getRadioButtonGroups()[0].getSelectedCheckbox()
    if not checkbox:
        return
    drawnRoiName = checkbox.label
    drawnRoi.setName(drawnRoiName)
    drawnRoi.setHandleSize(SIZE_HANDLE)
    index = get_roi_index_by_name(drawnRoiName) # None if the drawnRoi does not already exist

    if 'section-' in drawnRoiName:
        # the roi necessarily already exists
        drawnRoi.setStrokeColor(Color.blue)
        manager.setRoi(drawnRoi, index)

    elif 'roi-' in drawnRoiName:
        drawnRoi.setStrokeColor(Color.yellow)
        if index: # existing roi
            manager.setRoi(drawnRoi, index)
        else: # new roi
            manager.addRoi(drawnRoi)
            reorder_roi_manager()
            time.sleep(0.2)

    elif 'focus-' in drawnRoiName:
        drawnRoi.setStrokeColor(Color.green)
        if index: # existing roi
            manager.setRoi(drawnRoi, index)
        else: # new roi
            manager.addRoi(drawnRoi)
            reorder_roi_manager()
            time.sleep(0.2)

    elif 'magnet-' in drawnRoiName:
        poly = drawnRoi.getFloatPolygon()
        x,y = poly.xpoints[0], poly.ypoints[0] # in case there are two points
        newPointRoi = PointRoi(x,y)
        newPointRoi.setHandleSize(SIZE_HANDLE)
        newPointRoi.setName(drawnRoiName)
        waferIm.killRoi()
        newPointRoi.setStrokeColor(Color.yellow)
        newPointRoi.setImage(waferIm)
        newPointRoi.setPosition(waferIm.getSlice())
        if index: # existing roi
            manager.setRoi(newPointRoi, index)
        else: # new roi
            manager.addRoi(newPointRoi)
            reorder_roi_manager()
            time.sleep(0.2)

    # select the drawnRoi
    index = get_roi_index_by_name(drawnRoiName)
    manager.select(index)

    manager_to_magc_local()

    # # elif:
        # # delete a section only possible in global mode
        # # ask if want to add section or ROI
            # # if section then any associated ROI will be removed

        # # there is a roiSection in roimanager associated to this slice
            # # use magc or use ROIManager?
            # # magc is ok
        # # then this is a roi -> add/overwrite ROI

def handleKeypressLocalModeP():
    global magc
    manager = get_roi_manager()
    waferIm = IJ.getImage()
    manager.runCommand('Update') # to update the current ROI

    manager_to_magc_local() # needed?

    allAnnotationNames = [
        annotation.getName()
        for annotation in manager.getRoisAsArray()]

    selectedIndexes = manager.getSelectedIndexes()
    if len(selectedIndexes) != 1:
        return

    selectedAnnotation = manager.getRoi(selectedIndexes[0])
    annotationName = selectedAnnotation.getName()

    annotationType = get_annotation_type(annotationName)

    if annotationType in ['landmarkEM', 'section']:
        IJ.showMessage('Info',
            ('You just pressed [p] for the propagation tool. Select first a ROI, a magnet point, or focus point(s).'
            + '\nAfter closing this message you can press [h] for help.'))
        return

    sectionId = int(annotationName.split('-')[1])

    min_section_id = min(magc['sections'].keys())
    max_section_id = max(magc['sections'].keys())

    gd = GenericDialogPlus('Propagation')
    gd.addMessage(
        'This ' + annotationType + ' is defined in section number ' + str(sectionId) + '.\n'
        + 'To what sections do you want to propagate this ' + annotationType + '?')
    gd.addStringField(
        'Enter a range or single values separated by commas. '
        + 'Range can be start-end (4-7 = 4,5,6,7) or '
        + 'start-end-increment (2-11-3 = 2,5,8,11).',
        str(min_section_id) + '-' + str(max_section_id))
    gd.addButton(
        'All sections' + ' ' + str(min_section_id) + '-' + str(max_section_id),
        ButtonClick())
    gd.addButton(
        'First half of the sections' + ' ' + str(min_section_id) + '-' + str(int(max_section_id/2.)),
        ButtonClick())
    gd.addButton(
        'Every second section' + ' ' + str(min_section_id) + '-' + str(max_section_id) + '-' + str(2),
        ButtonClick())

    gd.showDialog()
    if not gd.wasOKed():
        return
    userRange = gd.getNextString()
    inputIndexes = get_indexes_from_user_string(userRange)
    IJ.log('User input indexes from Propagation Dialog: ' + str(inputIndexes))

    newRois = []

    filteredInputIndexes = [
        i for i in inputIndexes
        if i in magc['sections'].keys()]

    if filteredInputIndexes == []:
        return
    for inputIndex in filteredInputIndexes:
        if annotationType == 'roi':
            propagatedPoints = propagate_points(
                magc['sections'][sectionId]['polygon'],
                magc['rois'][sectionId]['polygon'],
                magc['sections'][inputIndex]['polygon'])

        elif annotationType == 'magnet': # magnet
            propagatedPoints = propagate_points(
                magc['sections'][sectionId]['polygon'],
                [magc['magnets'][sectionId]['location']],
                magc['sections'][inputIndex]['polygon'])

        elif annotationType == 'focus': # focus
            propagatedPoints = propagate_points(
                magc['sections'][sectionId]['polygon'],
                magc['focus'][sectionId]['polygon'],
                magc['sections'][inputIndex]['polygon'])

        # transform in local coordinates
        propagatedPoints = (
            transform_points_from_global_to_local(
                propagatedPoints,
                transformInfos[inputIndex]))
        if annotationType in ['roi', 'focus']:
            propagatedRoi = PolygonRoi(
                [p[0] for p in propagatedPoints],
                [p[1] for p in propagatedPoints],
                PolygonRoi.POLYGON)
        elif annotationType == 'magnet':
            propagatedRoi = PointRoi(
                [propagatedPoints[0][0]],
                [propagatedPoints[0][1]])
        else:
            return

        annotationName = annotationType + '-' + str(inputIndex).zfill(4)
        # # delete if existing roi
        # if annotationName in allannotationNames:
            # idRoiManager = allannotationNames.index(annotationName)
            # manager.select(idRoiManager)
            # time.sleep(0.01)
            # manager.runCommand('Delete')
            # time.sleep(0.01)
        propagatedRoi.setName(annotationName)
        propagatedRoi.setStrokeColor(
            Color.yellow if annotationType != 'focus'
            else Color.green)
        propagatedRoi.setHandleSize(SIZE_HANDLE)
        propagatedRoi.setImage(waferIm)

        # the position in the stack is the same
        # as the position from the section-xxxx
        annotation_position = (manager.getRoi(
            get_roi_index_by_name(
                'section-' + str(inputIndex).zfill(4)))
                .getPosition())
        propagatedRoi.setPosition(annotation_position)

        newRois.append(propagatedRoi)

    overwrite_rois(newRois)
    reorder_roi_manager()
    # select the first propagated roi
    select_roi_by_name(
        annotationType
        + '-'
        + str(filteredInputIndexes[0]).zfill(4))
    manager_to_magc_local()

    # update the templating info in general magc dictionary
    for id in filteredInputIndexes:
        annotationKey = (annotationType + 's').replace('ss','s')
        magc[annotationKey][id]['template'] = sectionId
    save_all_local_rois()

def handleKeypressLocalModeX():
    global magc # needed?
    manager = get_roi_manager()
    waferIm = IJ.getImage()

    allRoiNames = [roi.getName() for roi in manager.getRoisAsArray()]

    selectedIndexes = manager.getSelectedIndexes()
    if len(selectedIndexes) != 1:
        return
    selectedAnnotation = manager.getRoi(selectedIndexes[0])
    annotationName = selectedAnnotation.getName()

    annotationType = annotationType = get_annotation_type(annotationName)

    sectionId = int(annotationName.split('-')[1])
    stack_slice = selectedAnnotation.getPosition()

    if annotationType in ['magnet', 'roi']:
        if get_OK('Delete ' + annotationType + '-' + str(sectionId).zfill(4) + '?'):
            waferIm.killRoi()
            delete_roi(annotationName)
            reorder_roi_manager()
            manager_to_magc_local()

    if annotationType == 'section':
        sectionName = 'section-' + str(sectionId).zfill(4)
        roiName = 'roi-' + str(sectionId).zfill(4)
        magnetName = 'magnet-' + str(sectionId).zfill(4)
        focusName = 'focus-' + str(sectionId).zfill(4)

        message = 'Delete ' + annotationType + '-' + str(sectionId).zfill(4) + '?'

        # gather all existing annotations
        if any([x in allRoiNames
            for x in [roiName, magnetName, focusName]]):

            message = message + '\nIt will also delete '

            if roiName in allRoiNames:
                message += '\n' + roiName

            if magnetName in allRoiNames:
                message += '\n' + magnetName

            if focusName in allRoiNames:
                message += '\n' + focusName
            message += '.'

        if get_OK(message):

            if waferIm.getNSlices() == 1:
                if get_OK('Case not yet handled: you are trying to delete the only existing section.'
                 + '\n\nFiji will close. Please delete the .ini file and start over from scratch instead.\n\nContinue?'):
                    waferIm.close()
                    manager.close()
                    sys.exit()
                else:
                    return

            waferIm.killRoi()
            for name in [roiName, magnetName, focusName, sectionName]:
                delete_roi(name)
            # delete current slice
            waferIm.getImageStack().deleteSlice(stack_slice)
            newStack = waferIm.getImageStack()
            waferIm.setStack(
                newStack,
                1,
                newStack.getSize(),
                1)

            # clean manager
            reassign_roi_slices_after_section_deletion(stack_slice)
            reorder_roi_manager()
            manager_to_magc_local()

    # selecting the previous section roi
    while sectionId > -1:
        sectionName = 'section-' + str(sectionId).zfill(4)
        try:
            select_roi_by_name(sectionName)
            return
        except Exception as e:
            sectionId -=1
    manager.select(0)

def handleKeypressLocalMode(keyEvent):
    global magc
    keycode = keyEvent.getKeyCode()
    manager = get_roi_manager()
    waferIm = IJ.getImage()

    if keycode == KeyEvent.VK_A:
        handleKeypressLocalModeA()

    if keycode == KeyEvent.VK_M:
        handleKeypressLocalModeM()

    if keycode == KeyEvent.VK_X:
        handleKeypressLocalModeX()

    if keycode == KeyEvent.VK_P:
        handleKeypressLocalModeP()

    if keycode == KeyEvent.VK_Q:
        manager_to_magc_local()
        compute_save_tsp_order()
        close_local_mode()

    if keycode == KeyEvent.VK_D: # section down
        move_roi_manager_selection(-1)

    if keycode == KeyEvent.VK_F: # section up
        move_roi_manager_selection(1)

    if keycode == KeyEvent.VK_C: # 10 sections down
        move_roi_manager_selection(-10)

    if keycode == KeyEvent.VK_V: # 10 sections up
        move_roi_manager_selection(10)

    if keycode == KeyEvent.VK_E: # select first section
        selectedIndex = manager.getSelectedIndex()
        if selectedIndex != -1:
            manager.runCommand('Update')
        set_roi_and_update_roi_manager(0)

    if keycode == KeyEvent.VK_R: # select last section
        selectedIndex = manager.getSelectedIndex()
        if selectedIndex != -1:
            manager.runCommand('Update')
        set_roi_and_update_roi_manager(manager.getCount()-1)

    if keycode == KeyEvent.VK_G: # update drawing
        manager.runCommand('Update')
        manager_to_magc_local()

    if keycode == KeyEvent.VK_T: # toggle between local and global modes
        close_local_mode()
        IJ.log('Entering global mode...')
        start_global_mode()

    if keycode == KeyEvent.VK_S: # save
        save_all_local_rois()

    # compute TSP order
    if keycode == KeyEvent.VK_O:
        save_all_local_rois()
        handleKeypressO()
        save_all_local_rois()

    if keycode == KeyEvent.VK_Z: # terminate and save
        if globalMode:
            close_global_mode()
        else:
            close_local_mode()

    if keycode == KeyEvent.VK_H: # display help
        IJ.showMessage(
            'Help for local mode',
            helpMessageLocalMode)

    keyEvent.consume()

def addKeyListenerEverywhere(myListener):
    for elem in ([
        IJ.getImage().getWindow(),
        IJ.getImage().getWindow().getCanvas(),]
        # ui.getDefaultUI().getConsolePane().getComponent(),]
        # IJ.getInstance]
        + list(WindowManager.getAllNonImageWindows())):
        kls = elem.getKeyListeners()
        map(elem.removeKeyListener, kls)
        elem.addKeyListener(myListener)

    for id0,comp0 in enumerate(get_roi_manager().getComponents()):
        for kl in comp0.getKeyListeners():
            comp0.removeKeyListener(kl)
        comp0.addKeyListener(myListener)

        for id1,comp1 in enumerate(comp0.getComponents()):
            # if (type(comp1) == Button) and (comp1.getLabel() != 'Delete'):
            if type(comp1) == Button:
                comp0.remove(comp1)
            elif type(comp1) == Checkbox and not globalMode:
                comp0.remove(comp1)
            else:
                for kl in comp1.getKeyListeners():
                    comp1.removeKeyListener(kl)
                comp1.addKeyListener(myListener)
                try:
                    for id2,comp2 in  enumerate(comp1.getComponents()):
                        for kl in comp2.getKeyListeners():
                            comp2.removeKeyListener(kl)
                        comp2.addKeyListener(myListener)
                except:
                    pass

def addMouseWheelListenerEverywhere(myListener):
    for elem in ([
        IJ.getImage().getWindow(),
        IJ.getImage().getWindow().getCanvas(),]
        # ui.getDefaultUI().getConsolePane().getComponent(),]
        # IJ.getInstance]
        + list(WindowManager.getAllNonImageWindows())):
        kls = elem.getMouseWheelListeners()
        map(elem.removeMouseWheelListener, kls)
        elem.addMouseWheelListener(myListener)

    for id0,comp0 in enumerate(get_roi_manager().getComponents()):
        for kl in comp0.getMouseWheelListeners():
            comp0.removeMouseWheelListener(kl)
        comp0.addMouseWheelListener(myListener)

        for id1,comp1 in enumerate(comp0.getComponents()):
            # if (type(comp1) == Button) and (comp1.getLabel() != 'Delete'):
            if (type(comp1) == Button):
                comp0.remove(comp1)
            else:
                for kl in comp1.getMouseWheelListeners():
                    comp1.removeMouseWheelListener(kl)
                comp1.addMouseWheelListener(myListener)
                try:
                    for id2,comp2 in  enumerate(comp1.getComponents()):
                        for kl in comp2.getMouseWheelListeners():
                            comp2.removeMouseWheelListener(kl)
                        comp2.addMouseWheelListener(myListener)
                except:
                    pass

def set_roi_and_update_roi_manager(roiIndex, select = True):
    manager = get_roi_manager()
    nRois = manager.getCount()
    scrollPane = [
        component
        for component in manager.getComponents()
        if 'Scroll' in str(type(component))][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barMax = scrollBar.getMaximum()
    barWindow = scrollBar.getVisibleAmount()
    roisPerBar = nRois/float(barMax)
    roisPerWindow = barWindow * roisPerBar
    scrollValue = int((roiIndex - roisPerWindow/2.)/float(roisPerBar))
    scrollValue = max(0, scrollValue)
    scrollValue = min(scrollValue, barMax)
    if select:
        manager.select(roiIndex)
    # if nRois>roisPerWindow:
        # scrollBar.setValue(scrollValue)
    scrollBar.setValue(scrollValue)

def roi_manager_scroll_bottom():
    manager = get_roi_manager()
    nRois = manager.getCount()
    scrollPane = [
        component
        for component in manager.getComponents()
        if 'Scroll' in str(type(component))][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barMax = scrollBar.getMaximum()
    scrollBar.setValue(int(1.5 * barMax))

def get_central_roi_index():
    nRois = manager.getCount()
    scrollPane = [
        component for component
        in manager.getComponents()
        if 'Scroll' in str(type(component))][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barMax = scrollBar.getMaximum()
    barWindow = scrollBar.getVisibleAmount()
    roisPerBar = nRois/float(barMax)
    roisPerWindow = barWindow * roisPerBar
    barValue = scrollBar.getValue()
    barValue = min(barValue, barMax-barWindow)
    roiIndex = int(barValue * roisPerBar + roisPerWindow/2.)
    return roiIndex

def thread_select_window(im): # to select the window after showing the dialog
    time.sleep(0.2)
    # IJ.log('window selected')
    # WindowManager.setCurrentWindow(w)
    IJ.selectWindow(im.getTitle())

def thread_focus_on_OK(button):
    time.sleep(0.1)
    try:
        button.requestFocusInWindow()
    except Exception as e:
        pass

def focus_on_ok(dialog):
    ok_buttons = [
        button for button in list(dialog.getButtons())
        if button and button.getLabel()=='  OK  ']
    try:
        ok_button = ok_buttons[0]
        threading.Thread(
            target=thread_focus_on_OK,
            args=[ok_button]).start()
    except Exception as e:
        pass

def fill_section_dict(roi, compression = 1):
    d = {}
    points = points_from_poly(roi.getFloatPolygon())
    d['polygon'] = [
        [point[0], point[1]] for point in points]
    d['center'] = centroid(points)
    angle = get_angle([
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1]]) * 180 / Math.PI
    angle = (-angle)%(-360) + 180
    d['angle'] = round(angle, 3)
    d['area'] = polygon_area(points)
    d['compression'] = compression
    return d

def fill_roi_dict(roi, template=-1):
    d = {}
    d['template'] = template
    points = points_from_poly(roi.getFloatPolygon())
    d['polygon'] = [
        [point[0], point[1]] for point in points]
    d['center'] = centroid(points)
    angle = get_angle([
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1]]) * 180 / Math.PI
    angle = (-angle)%(-360) + 180
    d['angle'] = round(angle, 3)
    d['area'] = polygon_area(points)
    return d

def get_roi_index_by_name(name):
    manager = get_roi_manager()
    try:
        index = [manager.getName(i)
            for i in range(manager.getCount())].index(name)
        return index
    except Exception as e:
        return None

def manager_to_magc_global():
    global magc
    manager = get_roi_manager()
    annotations = manager.getRoisAsArray()

    # collect things before recreating magc
    serial_order = copy.deepcopy(magc['serialorder'])
    templated_annotations = collect_templated_annotations(magc)
    try:
        tsp_order = copy.deepcopy(magc['tsporder'])
    except Exception as e:
        tsp_order = []

    magc = create_empty_magc()

    magc['serialorder'] = serial_order
    magc['tsporder'] = tsp_order

    for annotation in annotations:
        annotationName = annotation.getName()
        annotationId = int(annotationName.split('-')[-1])
        annotationType = get_annotation_type(annotationName)

        if annotationType == 'section':
            magc['sections'][annotationId] = fill_section_dict(annotation)

        elif annotationType == 'roi':
            magc['rois'][annotationId] = fill_roi_dict(annotation)

        elif annotationType in ['landmark_em', 'magnet', 'focus']:
            points = points_from_poly(annotation.getFloatPolygon())

            if annotationType == 'landmark_em':
                magc['landmarksEM'][annotationId] = points[0]

            elif annotationType == 'magnet':
                magc['magnets'][annotationId] = {
                    'template': -1,
                    'location': points[0]
                    }
            elif annotationType == 'focus':
                magc['focus'][annotationId] = {
                    'template': -1,
                    'polygon': points}

    # re-populate templated annotations
    for key,val in templated_annotations.iteritems():
        annotationType, id = key.split('_')
        id = int(id)
        annotationKey = (annotationType + 's').replace('ss','s')
        magc[annotationKey][id]['template'] = val[0]

    # IJ.log('*** managerToMagCGlocal ***' + str(magc))

def transform_points_from_local_to_global(points, transformInfo):
    id, key, _, _, _, _, rotateTransform, translationTransform = transformInfo
    points1 = apply_transform(points, translationTransform.createInverse())
    points2 = apply_transform(points1, rotateTransform.createInverse())
    return points2

def transform_points_from_global_to_local(points, transformInfo):
    id, key, _, _,_,_, rotateTransform, translationTransform = transformInfo
    points1 = apply_transform(points, rotateTransform)
    points2 = apply_transform(points1, translationTransform)
    return points2

def points_from_poly(poly):
    xPoly = poly.xpoints
    yPoly = poly.ypoints
    return [[x,y] for x,y in zip(xPoly, yPoly)]

def manager_to_magc_local():
    global magc
    manager = get_roi_manager()
    annotations = manager.getRoisAsArray()

    serial_order = copy.deepcopy(magc['serialorder'])
    tsp_order = copy.deepcopy(magc['tsporder'])
    landmarksEM = copy.deepcopy(magc['landmarksEM'])

    # collect templating information before rebuilding

    templated_annotations = collect_templated_annotations(magc)
    # IJ.log('templated_annotations' + str(templated_annotations))

    changed_sections = get_changed_sections(annotations, magc)

    magc = create_empty_magc()

    magc['serialorder'] = serial_order
    magc['tsporder'] = tsp_order
    magc['landmarksEM'] = landmarksEM

    for annotation in annotations:
        annotationName = annotation.getName()
        annotationId = int(annotationName.split('-')[-1])
        annotationType = get_annotation_type(annotationName)

        points = points_from_poly(annotation.getFloatPolygon())

        transformedPoints = (
            transform_points_from_local_to_global(
                points,
                transformInfos[annotationId]))

        if annotationType == 'magnet':
            # globalRoi = PointRoi(*transformedPoints[0]) # useless I believe
            # globalRoi.setName(annotationName) # useless I believe
            magc['magnets'][annotationId] = {
                'template': -1, # will be updated at the end
                'location': transformedPoints[0]}

            # check whether the annotation was templated,
            # and if yes, whether it has changed
            # if yes: the templating is reverted to -1
            # if no: the templating reference is maintained
            if not annotationId in changed_sections:
                try:
                    templated_key = annotationType + '_' + str(annotationId)
                    location = templated_annotations[templated_key][1]['location']
                    distance = Point2D.distance(
                        location[0],
                        location[1],
                        transformedPoints[0][0],
                        transformedPoints[0][1])

                    if are_points_different([location, transformedPoints[0]]):
                        magc['magnets'][annotationId]['template'] = -1
                    else:
                        magc['magnets'][annotationId]['template'] = templated_annotations[templated_key][0]
                except Exception as e:
                    pass

        else:
            globalRoi = PolygonRoi(
                [point[0] for point in transformedPoints],
                [point[1] for point in transformedPoints],
                Roi.POLYGON)
            # globalRoi.setName(annotationName) # useless I believe

            if annotationType == 'section':
                magc['sections'][annotationId] = fill_section_dict(globalRoi)
            elif annotationType == 'roi':
                magc['rois'][annotationId] = fill_roi_dict(globalRoi)

                # check whether the annotation was templated,
                # and if yes, whether it has changed
                # if yes: the templating is reverted to -1
                # if no: the templating reference is maintained
                if not annotationId in changed_sections:
                    templated_key = annotationType + '_' + str(annotationId)
                    if templated_key in templated_annotations:
                        polygon = templated_annotations[templated_key][1]['polygon']
                        if are_polygons_different(polygon, transformedPoints):
                            magc['rois'][annotationId]['template'] = -1
                        else:
                            magc['rois'][annotationId]['template'] = templated_annotations[templated_key][0]

            elif annotationType == 'focus':
                magc['focus'][annotationId] = {
                    'template': -1,
                    'polygon': transformedPoints}

                # check whether the annotation was templated,
                # and if yes, whether it has changed
                # if yes: the templating is reverted to -1
                # if no: the templating reference is maintained
                if not annotationId in changed_sections:
                    templated_key = annotationType + '_' + str(annotationId)
                    if templated_key in templated_annotations:
                        polygon = templated_annotations[templated_key][1]['polygon']
                        if are_polygons_different(polygon, transformedPoints):
                            magc['focus'][annotationId]['template'] = -1
                        else:
                            magc['focus'][annotationId]['template'] = templated_annotations[templated_key][0]

    # # # # re-populate templated annotations
    # # # for annotationType_id in templated_annotations:
        # # # annotationType, id = annotationType_id.split('_')
        # # # id = int(id)
        # # # magc[annotationType][id]['template'] = annotationType_id[0]


def save_all_global_rois():
    manager_to_magc_global()
    write_ini(magc)
    IJ.log(str(datetime.datetime.now()) + ' - Everything saved in ' + iniPath)

def save_all_local_rois():
    manager_to_magc_local()
    write_ini(magc)
    IJ.log(str(datetime.datetime.now()) + ' - Everything saved in ' + iniPath)

# def toggleLocalGlobalModes():
    # if globalMode:
        # IJ.log('Entering local mode...')
        # # if :
            # # IJ.log('Cannot enter local mode, there are no sections')
            # # IJ.showMessage('Warning', 'Some sections lack one part (tissue or magnet) and will not be displayed during local')
    # else:
        # close_local_mode()
        # IJ.log('Entering global mode...')
        # start_global_mode()

def sections_sanity_checks(sections):
    # sanity checks
    sectionSet = set(sections['sections'].keys())
    roiSet = set(sections['rois'].keys())
    if len(sectionSet) == 0:
        IJ.log('There are no sections')
        return False
    # if not tissueSet == magnetSet:
        # IJ.log('Missing Magnet parts ' + str(tissueSet - magnetSet))
        # IJ.log('Missing Tissue parts ' + str(magnetSet - tissueSet))
        # return False
    # if len(sections['tissue'].keys()) != len(sections['magnet'].keys()):
        # IJ.log('There are duplicates somewhere:')
        # IJ.log('Tissue duplicates: ' + str(set([x for x in sections['tissue'].keys() if l.count(x) > 1])))
        # IJ.log('Magnet duplicates: ', set([x for x in sections['magnet'].keys() if l.count(x) > 1]))
        # return False
    return True

'''
def get_display_parameters(magc):
    # calculate [displaySize, cropSize, tissueMagnetDistance] based on sectionSize
    sectionExtent = 0

    for section in magc['sections'].values():
        sectionPoints = section['polygon']
        sectionExtent = max(sectionExtent, longest_diagonal(sectionPoints))

    displaySize = [int(1.2 * sectionExtent), int(1.2 * sectionExtent)]
    displaySectionCenter = [int(0.5*displaySize[0]), int(0.5*displaySize[1])]
    cropSize = [2*displaySize[0], 2*displaySize[1]]
    return displaySectionCenter, displaySize, cropSize
'''

def get_display_parameters(magc):
    # calculate [displaySize, cropSize, tissueMagnetDistance] based on sectionSize

    if magc['magnets'] == {}:
        sectionExtent = 0

        for section in magc['sections'].values():
            sectionPoints = section['polygon']
            sectionExtent = max(sectionExtent, longest_diagonal(sectionPoints))

        displaySize = [int(1.2 * sectionExtent), int(1.2 * sectionExtent)]
        displaySectionCenter = [int(0.5*displaySize[0]), int(0.5*displaySize[1])]
        cropSize = [2*displaySize[0], 2*displaySize[1]]
        return displaySectionCenter, 0, displaySize, cropSize

    else:
        sectionExtent = 0
        magnetExtent = 0
        for section in magc['sections'].values():
            sectionPoints = section['polygon']
            sectionExtent = max(sectionExtent, longest_diagonal(sectionPoints))

        # for magnet in magc['magnets'].values():
            # magnetPoints = magnet['polygon']
            # magnetExtent = max(magnetExtent, longest_diagonal(magnetPoints))

        tissueMagnetDistances = []
        for key, _ in magc['magnets'].iteritems():
            sectionCenter = magc['sections'][key]['center']
            magnetCenter = magc['magnets'][key]['location']
            tissueMagnetDistances.append(get_distance(sectionCenter, magnetCenter))

        if tissueMagnetDistances != []:
            tissueMagnetDistance = sum(tissueMagnetDistances)/len(tissueMagnetDistances)

        # displaySize = [int(1.2 * max(sectionExtent, magnetExtent)), int(0.7 * (sectionExtent + magnetExtent))]
        displaySize = [int(1.2 * sectionExtent), int(2*0.7*sectionExtent)]
        displayTissueCenter = [(displaySize[0]/2), (displaySize[1] - 1.2 * (sectionExtent/2))]

        cropSize = [2*displaySize[0], 2*displaySize[1]]
        return displayTissueCenter, tissueMagnetDistance, displaySize, cropSize

def init_manager():
    # get, place, and reset the ROIManager
    manager = get_roi_manager()
    manager.setTitle('Annotations')
    manager.reset()
    manager.setSize(250, int(0.95*screenSize.height)) # 280 so that the title is not cut
    manager.setLocation(screenSize.width - manager.getSize().width, 0)
    return manager

def get_roi_manager():
    manager = RoiManager.getInstance()
    if manager == None:
        manager = RoiManager()
    return manager

def start_local_mode():
    global globalMode
    globalMode = False
    #############################
    # get sections and parameters
    global magc # needs to be accessible outside
    magc = read_ini()
    if not sections_sanity_checks(magc):
        IJ.showMessage('Something is wrong with the sections in the .magc file (probably no section defined). Fix it with the global mode.')
        start_global_mode()
        return

    displaySectionCenter, tissueMagnetDistance, displaySize, cropSize = get_display_parameters(magc)
    manager = init_manager()
    #############################

    ####################################################################
    # compute transforms (transformInfos) and show the stack of sections
    waferIm = IJ.openImage(waferImPath)
    ims = ImageStack(displaySize[0], displaySize[1])
    global transformInfos
    transformInfos = {}
    for id, key in enumerate(magc['sections'].keys()):
        sectionPoints = magc['sections'][key]['polygon']
        sectionCenter = magc['sections'][key]['center']
        sectionAngle = magc['sections'][key]['angle']
        roiPoints = []
        if key in magc['rois'].keys():
            roiPoints = magc['rois'][key]['polygon']
        else:
            roiPoints = []

        if key in magc['magnets'].keys():
            magnetPoint = magc['magnets'][key]['location']
        else:
            magnetPoint = []

        if key in magc['focus'].keys():
            focusPoints = magc['focus'][key]['polygon']
        else:
            focusPoints = []

        translation = [
            displaySectionCenter[0] - sectionCenter[0],
            displaySectionCenter[1] - sectionCenter[1]]

        cropBoxBeforeRotation = Roi(
            sectionCenter[0] - cropSize[0]/2,
            sectionCenter[1] - cropSize[1]/2,
            cropSize[0],
            cropSize[1])

        croppedSectionGlobal = crop(waferIm, cropBoxBeforeRotation)
        rotate(croppedSectionGlobal, sectionAngle)
        rotatedCropped = croppedSectionGlobal

        cropBoxForDisplay = Roi(
            (cropSize[0] - displaySize[0])/2.,
            (cropSize[1] - displaySize[1])/2. - (tissueMagnetDistance/2.),
            displaySize[0],
            displaySize[1])

        croppedRotatedCropped = crop(rotatedCropped, cropBoxForDisplay)
        croppedRotatedCropped.setTitle('section-' + str(key).zfill(4))

        # rotation around sectionCenter
        rotateTransform = AffineTransform.getRotateInstance(
            sectionAngle * Math.PI/180,
            sectionCenter[0],
            sectionCenter[1])

        # translation for displaySize cropping
        translationTransform = AffineTransform.getTranslateInstance(
            -(sectionCenter[0] - displaySize[0]/2.),
            -(sectionCenter[1] - displaySize[1]/2. - tissueMagnetDistance/2.))

        transformInfos[key] = [id, key, sectionPoints, roiPoints, magnetPoint,
            focusPoints, rotateTransform, translationTransform]

        ims.addSlice(
            croppedRotatedCropped.getTitle(),
            croppedRotatedCropped.getProcessor())

    imp = ImagePlus('Stack of aligned sections', ims)
    ####################################################################

    ##################################################
    # calculate the new ROI locations and add to manager
    for transformInfo in transformInfos.values():
        (id, key, sectionPoints, roiPoints,
        magnetPoint, focusPoints, rotateTransform,
        translationTransform) = transformInfo

        for annotationType, points, color in zip(
            ['section', 'roi', 'magnet', 'focus'],
            [sectionPoints, roiPoints, [magnetPoint], focusPoints],
            [Color.blue, Color.yellow, Color.yellow, Color.green]):

            if points and (points != [[]]): # in case there is no ROI, no magnet, or no focus
                rotatedPoints = apply_transform(points, rotateTransform) # rotate around sectionCenter
                translatedrotatedPoints = apply_transform(rotatedPoints, translationTransform) # translate for displaySize Cropping

                xPoly = [point[0] for point in translatedrotatedPoints]
                yPoly = [point[1] for point in translatedrotatedPoints]
                if len(xPoly)>1 or annotationType=='focus':
                    poly = PolygonRoi(xPoly, yPoly, PolygonRoi.POLYGON)
                else:
                    poly = PointRoi(xPoly[0], yPoly[0])
                poly.setName(annotationType + '-' + str(key).zfill(4))

                poly.setStrokeColor(color)
                poly.setImage(imp)
                poly.setPosition(id + 1)
                poly.setHandleSize(SIZE_HANDLE)
                manager.addRoi(poly)
    ##################################################

    #################
    # arrange windows
    IJInstance.setLocation(0, screenSize.height - IJInstance.getSize().height)
    # enlarge window, select first ROI, get polygon tool
    imp.show()
    win = imp.getWindow()
    if imp.getNSlices() >1:
        win.remove(win.getComponents()[1]) # remove slider bar
    win.maximize()
    win.setLocation(
        screenSize.width - manager.getSize().width - win.getSize().width,
        0)
    #################

    ######################################################
    # add listeners window, canvas and ROImanager
    localModeKeyListener = ListenToKey()
    addKeyListenerEverywhere(localModeKeyListener)

    mouseWheelListener = ListenToMouseWheel()
    addMouseWheelListenerEverywhere(mouseWheelListener)
    ######################################################
    # change the handle size for easy grabbing
    for roi in manager.getRoisAsArray():
        roi.setHandleSize(15)

    #######################
    # arrange windows again
    reorder_roi_manager()
    set_roi_and_update_roi_manager(0) # select first ROI

    IJ.setTool('polygon')
    manager.toFront()
    time.sleep(0.2)
    IJ.selectWindow(imp.getTitle())
    #######################

def close_local_mode():
    # if there is a ROI selection, then update the last user modification
    manager = get_roi_manager()
    if manager.getSelectedIndex() != -1:
        manager.runCommand('Update')

    save_all_local_rois()

    windowIds = WindowManager.getIDList()
    for windowId in windowIds:
        im = WindowManager.getImage(windowId)
        im.changes = False # to prevent a dialog
        im.close()

    manager = RoiManager.getInstance()
    if not manager == None:
        manager.reset()
        manager.close()

    # map(IJInstance.addKeyListener, initialIJKeyListeners) # restore the keylisteners of the IJ window

def start_global_mode(check_pad=False):
    global globalMode
    globalMode = True
    global magc
    magc = read_ini()

    if check_pad:
        waferIm = load_pad_save(
            waferImPath,
            PADDING_FACTOR)
    else:
        waferIm = IJ.openImage(waferImPath)
    waferIm.show()
    waferIm.getWindow().maximize()

    manager = init_manager()

    # populate the manager and draw on wafer picture
    for label in ['sections', 'rois']:
        for key in magc[label].keys():
            poly = polygonroi_from_points(magc[label][key]['polygon'])
            poly.setName(label[:-1] + '-' + str(key).zfill(4)) # to remove the last 's'
            c = Color.blue if label=='sections' else Color.yellow
            poly.setStrokeColor(c)
            poly.setImage(waferIm)
            poly.setHandleSize(SIZE_HANDLE)
            manager.addRoi(poly)

    for label in ['landmarksEM', 'magnets']:
        if magc[label] != {}:
            for key in magc[label]:
                if label=='magnets':
                    point = magc[label][key]['location']
                else:
                    point = magc[label][key]
                pointRoi = PointRoi(*point)
                pointRoi.setName(label
                    .replace('marks', 'mark')
                    .replace('magnets', 'magnet')
                    .replace('EM', '_em')
                    + '-'
                    + str(key).zfill(4))
                pointRoi.setImage(waferIm)
                pointRoi.setHandleSize(SIZE_HANDLE)
                manager.addRoi(pointRoi)

    for key in magc['focus'].keys():
        poly = polygonroi_from_points(magc['focus'][key]['polygon'])
        poly.setName('focus-' + str(key).zfill(4))
        poly.setStrokeColor(Color.green)
        poly.setImage(waferIm)
        poly.setHandleSize(SIZE_HANDLE)
        manager.addRoi(poly)

    manager.runCommand('UseNames', 'true')
    manager.runCommand('Show All with labels')
    reorder_roi_manager()
    IJ.setTool('polygon')
    manager.toFront()
    time.sleep(0.2)
    IJ.selectWindow(waferIm.getTitle())

    ######################################################
    # add keylistener to window and canvas of the picture,
    # and to IJInstance) for ROI navigation
    globalModeListener = ListenToKey()
    addKeyListenerEverywhere(globalModeListener)

    mouseWheelListener = ListenToMouseWheel()
    addMouseWheelListenerEverywhere(mouseWheelListener)

    ######################################################

def close_global_mode():
    save_all_global_rois()

    windowIds = WindowManager.getIDList()
    for windowId in windowIds:
        im = WindowManager.getImage(windowId)
        im.changes = False # to prevent a dialog
        im.close()

    manager = RoiManager.getInstance()
    if not manager == None:
        manager.reset()
        manager.close()

    map(IJInstance.addKeyListener, initialIJKeyListeners) # restore the keylisteners of the IJ window

def toggle_fill(label='section'):
    manager = get_roi_manager()
    rois = manager.getRoisAsArray()
    for roi in rois:
        if label in roi.getName():
            currentColor = roi.getFillColor()
            if currentColor:
                roi.setFillColor(None)
            else:
                color = Color.blue if label=='section' else Color.yellow
                roi.setFillColor(color)

    id = manager.getSelectedIndex()
    scrollPane = [
        component for component
        in manager.getComponents()
        if 'Scroll' in str(type(component))][0]
    scrollBar = scrollPane.getVerticalScrollBar()
    barValue = scrollBar.getValue()
    if id > -1:
        manager.select(id)
    else:
        manager.select(0)
        manager.runCommand('Deselect')
        scrollBar.setValue(barValue)
    manager.runCommand('Show All without labels')

def collect_templated_annotations(d):
    templated_annotations = {}
    for annotationKey, annotation_dic in d.iteritems():
        if annotationKey in ['rois', 'magnets', 'focus']:
            for id,dic in annotation_dic.iteritems():
                if 'template' in dic:
                    if dic['template'] != -1:
                        annotationType = 'focus' if ('ocus' in annotationKey) else annotationKey[:-1]
                        templated_annotations[
                            annotationType + '_'
                            + str(id)] = [
                                dic['template'],
                                dic]
    return templated_annotations

def get_changed_sections(annotations, magc):
    changed_sections = []
    for annotation in annotations:
        annotationName = annotation.getName()
        annotationId = int(annotationName.split('-')[-1])
        annotationType = get_annotation_type(annotationName)
        if annotationType=='section':
            points = points_from_poly(annotation.getFloatPolygon())
            transformedPoints = (
                transform_points_from_local_to_global(
                    points,
                    transformInfos[annotationId]))
            polygon = magc['sections'][annotationId]['polygon']
            if are_polygons_different(polygon, transformedPoints):
                changed_sections.append(annotationId)
    return changed_sections

# def are_points_different(p1, p2):
def are_points_different(points):
    p1, p2 = points
    distance = Point2D.distance(
        p1[0], p1[1],
        p2[0], p2[1])
    return distance > CHANGE_THRESHOLD

def are_polygons_different(poly1,poly2):
    if len(poly1)==len(poly2):
        return any(map(
            are_points_different,
            zip(poly1, poly2)))
    else:
        return True

##############
# Script start
##############

#################
# Initializations
IJ.log('Started.')

# ####################################################
# get concorde solver, linkern solver, and cygwin1.dll
# try to download if not present
pluginFolder = IJ.getDirectory('plugins')
cygwindllPath = os.path.join(
    pluginFolder,
    'cygwin1.dll')
linkernPath = os.path.join(
    pluginFolder,
    'linkern.exe')
concordePath = os.path.join(
    pluginFolder,
    'concorde.exe')

# download cygwin1.dll
if not os.path.isfile(cygwindllPath):
    # download cygwin1.dll
    cygwindll_url = r'https://raw.githubusercontent.com/templiert/MagFinder/master/cygwin1.dll'
    try:
        IJ.log('Downloading Windows-10 precompiled cygwin1.dll from ' + cygwindll_url)
        f = File(cygwindllPath)
        FileUtils.copyURLToFile(
            URL(cygwindll_url),
            f)
    except (Exception, java_exception) as e:
        IJ.log('Failed to download cygwin1.dll due to ' + str(e))

# download concorde and linkern solvers
if not os.path.isfile(concordePath):
    download_unzip(
        r'https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/cygwin/concorde.exe.gz',
        concordePath)
if not os.path.isfile(linkernPath):
    download_unzip(
        r'https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/cygwin/linkern.exe.gz',
        linkernPath)

# if one of the three files is missing
if not (os.path.isfile(cygwindllPath)
    and os.path.isfile(concordePath)
    and os.path.isfile(linkernPath)):
    concordePath = None
    linkernPath = None
    cygwindllPath = None
# ####################################################

# size of point to grab annotations
SIZE_HANDLE = 15
PADDING_FACTOR = 0.15

# distance to consider that an annotation has
# been changed (for templating)
CHANGE_THRESHOLD = 0.1

# image sizeof each sub-picture for the overview montage
LOCAL_SIZE_STANDARD = 400

ANNOTATION_TYPES = [
    'section',
    'roi',
    'magnet',
    'focus',
    'landmark_em']

try:
    experimentFolder = os.path.normpath(
        DirectoryChooser('Select the experiment folder.')
        .getDirectory())
except Exception as e:
    IJ.showMessage(
    'Exit',
    'There was a problem accessing the folder')
    sys.exit('No directory was selected. Exiting.')

if not os.path.isdir(experimentFolder):
    IJ.showMessage(
        'Exit',
        'No directory was selected. Exiting.')
    sys.exit('No directory was selected. Exiting.')

waferImNames = [
    imName for imName
    in os.listdir(experimentFolder)
    if (
        (os.path.splitext(imName)[1] in ['.tif', '.png', '.jpg'])
        and not 'verview' in imName)]

if not waferImNames:
    IJ.showMessage(
        'Message',
        ('There is no image (.tif, .png, .jpg) in the experiment folder you selected.'
        + '\n'
        + 'Add an image and start the plugin again.'))
    IJ.log('There is no image in the experiment folder')
    sys.exit()
else:
    waferImPath = os.path.join(
        experimentFolder,
        waferImNames[0])

# start in global mode
globalModeListener = ListenToKey()
globalMode = True # localMode or globalMode

IJInstance = IJ.getInstance()
initialIJKeyListeners = IJInstance.getKeyListeners()
screenSize = IJ.getScreenSize()

manager = get_roi_manager()

helpMessageLocalMode = (
    '<html><br><br><br><br><br><br><br><br>'
    + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
    + '<br><br> <h3>Navigation</h3>'
    + '<br><br><ul><li>'
    + 'Press [d]/[f] to navigate annotations up/down.'
    + '<br><br><li>'
    + 'Press [c]/[v] to navigate 10 annotations up/down.'
    + '<br><br><li>'
    + 'Press [e]/[r] to move to first/last annotation.'
    + '<br><br><li>'
    + 'If you lose the current annotation (by clicking outside of the annotation), then press [d],[f] or use the mouse wheel to make the annotation appear again.'
    + '<br><br><li>'
    + 'You can navigate the annotations with the mouse wheel. Holding [Ctrl] makes the mouse scrolling faster every 10 annotations.'
    + '</ul>'

    + '<br><br> <h3>Actions</h3>'
    + '<br><br><ul><li>'
    + 'Press [a] to create/modify an annotation.'
    + '<br><br><li>'
    + 'Press [t] to toggle to global mode.'
    + '<br><br><li>'
    + 'Press [g] to validate your modification. Validating your modification happens automatically when browsing through the sections with [d]/[f], [c]/[v], [e]/[r], or with the mouse wheel'
    + '<br><br><li>'
    + 'Press [q] to terminate. Everything will be saved.'
    + '<br><br><li>'
    + 'Press [s] to save your modifications to file (happens already automatically when toggling [t] or terminating [q]).'
    + '<br><br><li>'
    + 'Press [m] to export a summary montage. It is saved in the same folder as the wafer image.'
    + '<br><br><li>'
    + 'Press [o] to compute the section order that minimizes the travel of the microscope stage (not to be confused with the serial sectioning order of the sections).'
    +' The order is saved in the field "tsporder" (stands for Traveling Salesman Problem order)'
    + '<br><br><br><br><br><br></ul>')

helpMessageGlobalMode = (
    '<html><br><br><br><br><br><br>'
    + '<p style="text-align:center"><a href="https://youtu.be/ZQLTBbM6dMA">20-minute video tutorial</a></p>'
    + '<br><br> <h3>Navigation</h3>'
    + '<br><br><ul><li>'
    + 'Use the mouse wheel with the [Ctrl] key pressed to zoom in/out.'
    + '<br><br><li>'
    + 'Use the mouse wheel to navigate up/down.'
    + '<br><br><li>'
    + 'Use the mouse wheel with the [Shift] key pressed to navigate left/right.'
    + '<br><br><li>'
    + 'Press [g] to toggle painting of the sections.'
    + '<br><br><li>'
    + 'Press [b] to toggle painting of the ROIs.'
    + '</ul>'

    + '<br><br> <h3>Actions</h3>'
    + '<br><br><ul><li>'
    + 'Press [a] to add an annotation that you have drawn.'
    + '<br><br><li>'
    +'Press [t] to toggle to local mode.'
    + '<br><br><li>'
    +'Press [q] to terminate. Everything will be saved.'
    + '<br><br><li>'
    +'Press [s] to save (happens already automatically when toggling [t] and terminating [q]).'
    + '<br><br><li>'
    +'Press [m] to export a summary image. It is saved in the same folder as the wafer image.'
    + '<br><br><li>'
    +'Press [o] to compute the section order that minimizes the travel of the microscope stage (not to be confused with the serial sectioning order of the sections).'
    +' The order is saved in the field "tsporder" (stands for Traveling Salesman Problem order)'
    + '<br><br><br><br><br><br></ul>')
################

start_global_mode(check_pad=True)