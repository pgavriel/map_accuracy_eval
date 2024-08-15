#!/usr/bin/env python3
import sys
import cv2 as cv
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from os import listdir
from os.path import isfile, join
import csv
import datetime

'''
THIS FILE ASSUMES THE NEW CSV DATA FORMAT FOR DEFINED POINTS
WORK IN PROGRESS
'''

position_changed = True
pts_selection = 1
marker_position = 1
pts1 = dict()
pts1 = {1: [242, 157], 2: [838, 152], 3: [233, 726], 4: [845, 733]}
pts2 = dict()
pts2 = {1: [235, 581], 2: [234, 278], 3: [509, 564], 4: [499, 294], 5:[0,0]}

def click_event(event,x,y,flags,param):
    global marker_position, position_changed, pts_selection
    
    if event == cv.EVENT_LBUTTONUP:
        if (pts_selection == 1):
            pts1[marker_position] = [x, y]
            print("PTS1",pts1)
        elif (pts_selection == 2):
            pts2[marker_position] = [x, y] 
            print("PTS2",pts2)

        marker_position = marker_position + 1   
        position_changed = True


# def get_map_label(pts_file):
#     '''
#     Loads a csv textfile containing a maps reference points and
#     returns the first line of the file as the "label" for the map
#     '''
#     print("Getting label for "+str(pts_file)+" ...")
#     label = None
#     try:
#         f = open(pts_file,"r")
#         lines = f.readlines()
#         label = lines[0].split(',')[0]
    
#         print(f"Map Label: {label}")
#     except:
#         print("GET LABEL FAILED")
    
#     return label

# def import_reference_pts(pts_file):
#     '''
#     Loads a csv textfile into a dictionary, skipping the first line
#     Line format should be "ref point label, x value, y value"
#     '''
#     point_dict = dict()
#     print("Importing "+str(pts_file)+" ...")
#     try:
#         f = open(pts_file,"r")
#         lines = f.readlines()
#         lines = lines[1:]
#         for line in lines:
#             data = line.split(',')
#             # print(data)
#             # marker = int(data[0])
#             marker = str(data[0])
#             x = float(data[1])
#             y = float(data[2][:-1])
#             coords = [x,y]
#             point_dict[marker] = coords
#         print("Imported reference point dict:")
#         print(point_dict)
#     except:
#         print("PTS IMPORT FAILED")
    
#     return point_dict

# def export_reference_pts(pts_dict,name,dir=""):
#     pass
#     try:
#         filename = name+".pts"
#         filepath = os.path.join(dir,filename)
#     except:
#         print("Error exporting points.")
#         return
#     f = open(filepath,"w+")
#     f.write(name+"\n")
#     for p in sorted(pts_dict.keys()):
#         write_str = str(p)+","+str(pts_dict[p][0])+","+str(pts_dict[p][1])+"\n"
#         f.write(write_str)
#     f.close()
#     print("POINT DICT EXPORTED TO " + filename)
#     return

    
# def get_scale_factor(pix,mm):
#     '''Set scaling factor by providing pixels and the equivalent distance in millimeters'''
#     global scale
#     print("\nSetting scale")
#     print(str(pix)+"px == "+str(mm)+"mm")
#     scale = float(mm) / float(pix)
#     print("Scaling factor = "+str(scale))
#     return scale



def calculateDistance(x1,y1,x2,y2,precision=None):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if precision is not None:
        dist = round(dist,precision)
    return dist


def get_midpoint(data, include_z=False):
    """
    Calculate the midpoint for the x, y, and optionally z fields in the data.

    :param data: List of dictionaries, each containing 'x' and 'y' fields, and optionally 'z' field.
    :param include_z: Boolean flag to indicate whether to calculate the midpoint for the 'z' field.
    :return: A tuple containing (x_mid, y_mid, z_mid) if include_z is True,
             otherwise (x_mid, y_mid).
    """
    if not data:
        return None
    
    x_sum = 0
    y_sum = 0
    z_sum = 0 if include_z else None
    
    for point in data:
        x_sum += point['x']
        y_sum += point['y']
        if include_z:
            z_sum += point['z']
    
    num_points = len(data)
    x_mid = x_sum / num_points
    y_mid = y_sum / num_points
    
    if include_z:
        z_mid = z_sum / num_points
        return (x_mid, y_mid, z_mid)
    else:
        return (x_mid, y_mid)


def get_bounds(data, include_z=False):
    """
    Calculate the minimum and maximum bounds for the x, y, and optionally z fields in the data.

    :param data: List of dictionaries, each containing 'x' and 'y' fields, and optionally 'z' field.
    :param include_z: Boolean flag to indicate whether to calculate bounds for the 'z' field.
    :return: A tuple containing (x_min, x_max, y_min, y_max, z_min, z_max) if include_z is True,
             otherwise (x_min, x_max, y_min, y_max).
    """
    if not data:
        print("ERROR: get_bounds no data received.")
        return None
    
    # Initialize the min and max bounds with the first point
    x_min = x_max = data[0]['x']
    y_min = y_max = data[0]['y']
    if include_z:
        z_min = z_max = data[0]['z']
    
    # Iterate over the data to find min and max bounds
    for point in data[1:]:
        x_min = min(x_min, point['x'])
        x_max = max(x_max, point['x'])
        y_min = min(y_min, point['y'])
        y_max = max(y_max, point['y'])
        if include_z:
            z_min = min(z_min, point['z'])
            z_max = max(z_max, point['z'])
    
    if include_z:
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    else:
        return (x_min, x_max, y_min, y_max)


def flip_points(data, flip_x, flip_y):
    """
    Invert data values in the x and/or y fields.
    """
    if flip_x and flip_y: print("Flipping X and Y values...")
    elif flip_x: print("Flipping X values...")
    elif flip_y: print("Flipping Y values...")
    for point in data:
        if flip_x: point['x'] = point['x'] * -1
        if flip_y: point['y'] = point['y'] * -1
    
    return data

# TRANSLATION FUNCTIONS =========================================================================
def translate_points(data, offset):
    """
    Translates a list of 2D points by a specified [x, y] offset.

    Parameters:
        data (list of dict): A list of dictionaries, where each dictionary represents a point with 'x', 'y', and possibly other fields.
        offset (list or tuple): A list or tuple of two values [x_offset, y_offset] representing the translation offset.

    Returns:
        list of dict: A new list of dictionaries representing the translated points, including all other fields.
    """
    x_offset, y_offset = offset

    # Translate each point in the data
    translated_data = []
    for point in data:
        # Create a copy of the original point to avoid modifying the input data
        translated_point = point.copy()
        
        # Apply the translation to the 'x' and 'y' fields
        translated_point['x'] += x_offset
        translated_point['y'] += y_offset

        # Add the translated point to the list
        translated_data.append(translated_point)

    return translated_data


def move_points_to(data, target):
    """
    Moves the points so that their midpoint is at the specified target coordinates.

    Parameters:
        data (list of dict): A list of dictionaries, where each dictionary represents a point with 'x' and 'y' fields.
        target (list or tuple): A list or tuple of two values representing the target coordinates [x_target, y_target].

    Returns:
        list of dict: A new list of dictionaries representing the moved points, including all other fields.
    """
    # Get the current midpoint of the data
    midpoint = get_midpoint(data)

    # Calculate the offset needed to move the midpoint to the target
    x_offset = target[0] - midpoint[0]
    y_offset = target[1] - midpoint[1]

    # Translate the points in 2D
    return translate_points(data, (x_offset, y_offset))

#TODO: REIMPLEMENT
def auto_align(truth_dict,eval_dict,verbose=True):
    '''
    Translates the eval_dict so that it's midpoint is aligned with
    the midpoint of the truth_dict
    Returns aligned_dict, [x_offset,y_offset]
    '''
    # print(verbose)
    verbose = True
    if verbose: print("AUTO ALIGN")
    align_dict = eval_dict.copy()

    # Find midpoints of each reference point dictionary
    test_midpoint = get_dict_midpoint(align_dict)
    if verbose: print("Evaluation Midpoint: ",test_midpoint)
    ref_midpoint = get_dict_midpoint(truth_dict)
    if verbose: print("Ground Truth Midpoint: ",ref_midpoint)

    # Determine midpoint offset
    xy_off = [ref_midpoint[0]-test_midpoint[0],ref_midpoint[1]-test_midpoint[1]]
    if verbose: print("Eval Midpoint Translation:",xy_off)
    
    # Perform the translation to align the midpoints
    align_dict = translate_dict(align_dict,xy_off)

    return align_dict, xy_off

# ROTATION FUNCTIONS ====================================================================================
def rotate(origin, point, angle):
    """"
    Rotate a point counterclockwise by a given angle around a given origin. 
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_points(data,degrees):
    """
    Rotates a set of points within a data object around it's midpoint by a specified number of degrees.
    """
    angle_rad = math.radians(degrees)
    midpoint = get_midpoint(data)
    rotated_data = []
    for point in data:
        # Create a copy of the original point to avoid modifying the input data
        rotated_point = point.copy()
        rx, ry = rotate(midpoint, (rotated_point['x'],rotated_point['y']), angle_rad)
        # Apply the rotation to the 'x' and 'y' fields
        rotated_point['x'] = rx
        rotated_point['y'] = ry

        # Add the translated point to the list
        rotated_data.append(rotated_point)
        
    return rotated_data 

# TODO: Need to be reimplement auto_rotate function (finds the optimal rotation, 
#       minimizing distance between point pairs of two data objects)
#       Reference point_matching.py for old implementation.


# SCALING FUNCTIONS ============================================================
def scale_points(data, scale_factor,midpoint=None,verbose=True):
    """
    Scale a set of data points by a given scale factor centered around the midpoint of the points.
    """
    if midpoint is None:
        midpoint = get_midpoint(data)
    # Scale Test Dict
    print("Scaling Point Dictionary by factor of {}".format(scale_factor))
    scaled_data = []
    for point in data:
        scaled_point = point.copy()
        scaled_point['x'] = midpoint[0] + scale_factor * (scaled_point['x'] - midpoint[0])
        scaled_point['y'] = midpoint[1] + scale_factor * (scaled_point['y'] - midpoint[1])

        scaled_data.append(scaled_point)
        if verbose:
            print("{} -> {}".format((point['x'],point['y']),(scaled_point['x'],scaled_point['y'])))
    return scaled_data

# TODO: Need to be reimplement auto_scale function (finds the optimal scale factor, 
#       minimizing distance between point pairs of two data objects)
#       Reference point_matching.py for old implementation.






# FOR TESTING
def modify_dict(point_dict, x=0):
    print("Modding...")
    mod_dict = point_dict.copy()
    dict_mp = get_dict_midpoint(mod_dict)
    delta = x
    dx = random.randint(-50,50)
    dy = random.randint(-50,50)
    mod_dict['RA1'][0] = mod_dict['RA1'][0] + dx
    mod_dict['RA1'][1] = mod_dict['RA1'][1] + dy
    dx = random.randint(-50,50)
    dy = random.randint(-50,50)
    mod_dict['LA3'][0] = mod_dict['LA3'][0] + dx
    mod_dict['LA3'][1] = mod_dict['LA3'][1] + dy
    # for k in mod_dict.keys():
    #     if k[0] == 'R':
    #         print (k)
    #         # Shift
    #         mod_dict[k][0] = mod_dict[k][0] - 0
    #         # Rotate
    #         angle_rad = math.radians(delta)
    #         # rx, ry = rotate(dict_mp, mod_dict[k], angle_rad)
    #         rx, ry = rotate(mod_dict["RC9"], mod_dict[k], angle_rad)
    #         mod_dict[k] = [float(rx),float(ry)]


    return mod_dict



def main(argv):
    global position_changed, marker_position, pts_selection, pts1, pts2
    blank = np.zeros([1000,1000,3],dtype=np.uint8)
    blank.fill(255)
    img = blank.copy()
    pts1_color = (0,255,0)
    pts2_color = (255,0,255)

    uni_scale = 50
    pts1 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_omron_sr.pts")
    pts1 = scale_dict(pts1,uni_scale)
    # bounds = get_dict_bounds(pts1)
    # pts1 = translate_dict(pts1,[0,-bounds[1][0]])
    # export_reference_pts(pts1,"nerve_basement_omron_scaled","D:/map_accuracy_eval/example")
    # pts2 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_spot_pixelcoord.txt")
    # pts2 = scale_dict(pts2,uni_scale)
    # pts2 = translate_dict(pts2,[0,1200])

    pts2 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_spot_pixelcoord.txt")
    # pts2 = scale_dict(pts2,uni_scale)
    # pts2 = translate_dict(pts2,[0,-bounds[1][0]])
    # pts2 = modify_dict(pts2)

    cv.namedWindow("Point Matching",cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("Point Matching", click_event)
    font = cv.FONT_HERSHEY_PLAIN
    running = True
    while running:

        if position_changed:
            img = blank.copy()
            for k in pts1:
                # print(pts1[k])
                img = cv.circle(img, [int(pts1[k][0]),int(pts1[k][1])], 2, pts1_color, 2)
                img = cv.putText(img,str(k),[int(pts1[k][0]),int(pts1[k][1])],font,1,(0,0,0),1,cv.LINE_AA)
            for k in pts2:
                # print(pts2[k])
                img = cv.circle(img, [int(pts2[k][0]),int(pts2[k][1])], 2, pts2_color, 2)
                img = cv.putText(img,str(k),[int(pts2[k][0]),int(pts2[k][1])],font,1,(0,0,0),1,cv.LINE_AA)
            
            img_text = "Edit Pts:"+str(pts_selection)+"/2   Pos:"+str(marker_position)+"(Press 3 to match)"
            org = (5,25)
            img = cv.putText(img,img_text,org,font,2,(0,0,0),2,cv.LINE_AA)

            position_changed = False

        # KEYBOARD CONTROLS
        k = cv.waitKey(100) & 0xFF
        if k == ord('1'): # Edit pts1
            pts_selection = 1
            marker_position = 1
            position_changed = True
        if k == ord('2'): # Edit pts2
            pts_selection = 2
            marker_position = 1
            position_changed = True
        # if k == ord('3'): # Match points
        #     # cv.namedWindow("Error",cv.WINDOW_AUTOSIZE)
        #     # error_px, error_img = calc_error(pts1, pts2, blank.copy())
        #     scaled_dict, scale, error_img = auto_scale(pts1, pts2, blank.copy())
        #     cv.imshow("Point Matching", error_img)
        #     cv.waitKey()
        #     rot_dict, min_rot, rot_img = auto_rotate(pts1, scaled_dict, blank.copy())
        #     cv.imshow("Point Matching", rot_img)
        #     cv.waitKey()
        if k == ord('4'): # TESTING
            mtd, med, coverage = calc_coverage(pts1,pts2)
            # print(mtd, med, coverage)
            point_errors, global_error, error_std = calc_error(mtd, med)
            # print(point_errors, global_error, error_std)
            # pts2, min_scale, _ = auto_scale(mtd,med)
            mtd, min_scale, _ = auto_scale(med,mtd)
            # print(pts2, min_scale)
            # pts2, offset = auto_align(mtd,pts2)
            position_changed = True

            point_errors, global_error, error_std = calc_error(mtd, pts2)
            generate_pointerror_contour_plot(mtd,pts2,point_errors,"D:/map_accuracy_eval/example/spot_map_image.png")
            # generate_contour_plot(mtd,pts2,point_errors)
        if k == ord('5'): 
            pts1 = translate_dict(pts1,[100,100])
            position_changed = True
        if k == ord('6'): 
            pts2, offset = auto_align(pts1,pts2)
            position_changed = True
        if k == ord('q'): # Exit program
            running = False
        
        
        # print(help(rotate))
        cv.imshow("Point Matching", img)
        
# TODO: Main test case should verify the functionality of all contained functions
if __name__ == "__main__":
    # main(sys.argv[1:])
    eval_dir = "C:/Users/nullp/Documents/MapEvaluations/"
    eval_files = [f for f in listdir(eval_dir) if isfile(join(eval_dir, f))] 
    mapnum = 0
    # Set Parameters
    # save_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output"
    save_dir = "C:/Users/nullp/Documents/MapEvaluations/output"
    truth_pts_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/devens-f15-gps-gt.txt"
    # eval_pts_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/devens-f15-gps-gt.txt"
    eval_pts_file = "C:/Users/nullp/Documents/MapEvaluations/teal-react-photo.csv"
    # eval_pts_file = os.path.join(eval_dir,eval_files[mapnum])
    # map_name = "Devens F15 Teal Eval2"
    map_name = get_map_label(eval_pts_file)
    map_name = "teal-react-photo"
    fig_note = ""
    excluded_std = 0
    metrics = []
    scale_metrics = []

    # Setup Reference Points
    pts1_scale = 1000000
    pts1 = import_reference_pts(truth_pts_file)
    if pts1_scale is not None:
        pts1 = scale_dict(pts1,pts1_scale)
    # bounds = get_dict_bounds(pts1)
    # pts1 = translate_dict(pts1,[0,-bounds[1][0]])

    pts2_scale = None
    pts2 = import_reference_pts(eval_pts_file)
    if pts2_scale is not None:
        pts2 = scale_dict(pts2,pts2_scale)
    # pts2 = scale_dict(pts2,1/100)
    # pts2 = translate_dict(pts2,[0,-bounds[1][0]])
    # pts2 = modify_dict(pts2)


    # Load map background image if applicable
    map_image = None
    map_img_scale = None
    if map_image is not None:
        map_image = cv.imread("C:/Users/nullp/Projects/map_accuracy_eval/input/420-rough-raster-inv.png")
        if map_img_scale is not None:
            map_image = scale_image_with_aspect_ratio(map_image,map_img_scale)


    # Calculate metrics for pts1 (truth) and pts2 (eval)
    print(f"\nPerforming Evaluation: {map_name}")
    mtd, med, coverage = calc_coverage(pts1,pts2)
    metrics = calc_error(mtd, pts2,precision=12)
    metrics.append(coverage)
    # NOTE: metrics = [point error dict, scale factors dict, global error, global err std dev, coverage]

    # Create Scale Factor Plot
    title = "{} Scale Factors - Excluding +/- {:.2f}stddev".format(map_name,excluded_std)
    filename = "{}_scalefactor_exclude{:02d}".format(map_name,int(excluded_std*10)) 
    save_file = os.path.join(save_dir,filename)
    scale_metrics = generate_scalefactor_plot(mtd,pts2,metrics,excludestd=excluded_std,title=title,save_file=save_file,image=map_image,
                        units="px",note=fig_note)
    # NOTE: scale_metrics = [mean, std dev, normed std dev]
    
    # Create Contour Plot
    filename = "{}_globalerrorcontour".format(map_name)  
    save_file = os.path.join(save_dir,filename)
    title = "{} Global Error Contour Plot".format(map_name)
    # generate_pointerror_contour_plot(mtd,pts2,metrics,image=map_image,title=title,save_file=save_file)
   
    # generate_scalefactor_plot(mtd,pts2,metrics,excludestd=excluded_std,title=title,#image=map_image,
    #                     units="px",note=fig_note)

    log_results = True
    if log_results:
        log_dir = "C:/Users/nullp/Projects/map_accuracy_eval"
        log_name = "map_results.csv"
        log_data = []
        now = datetime.datetime.now()
        now.strftime('%Y-%m-%d-%H:%M:%S')
        log_data = [now, map_name, fig_note] + metrics[2:] + scale_metrics
        log_to_csv(log_dir,log_name,log_data)



