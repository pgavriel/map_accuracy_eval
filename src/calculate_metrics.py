import os
import csv
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
import point_manip as pm

def log_to_csv(log_dir, log_name, data_values):
    log_path = os.path.join(log_dir, log_name)
    header = ['Timestamp', 'Map Name', 'Note',
             'global error', 'global err std dev', 'coverage',
             'scale mean', 'scale std dev', 'normalized scale std dev']  # Adjust the header columns as needed
    
    # Check if the directory exists, create it if it does not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_exists = os.path.isfile(log_path)
    
    # Open the file in append mode
    with open(log_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # If the file does not exist, write the header first
        if not file_exists:
            print("LOG CREATED...")
            writer.writerow(header)
        
        # Append the data values
        writer.writerow(data_values)
        print("LOG DATA ADDED...")

def calc_coverage(truth_dict,eval_dict,verbose=True):
    '''
    RETURNS [Matching Truth Dict, Matching Eval Dict, Coverage Score]
    '''
    # Make sure parameters are dictionaries
    if verbose: print("CALCULATING COVERAGE")
    if isinstance(truth_dict, dict):
        truth_dict_copy = truth_dict.copy()
    else:
        print("ERROR: truth_dict was not of type dictionary.")
        return -1
    if isinstance(eval_dict, dict):
        eval_dict_copy = eval_dict.copy()
    else:
        print("ERROR: eval_dict was not of type dictionary.")
        return -1

    # Make sure both dicts only contain matching entries
    for p in truth_dict:
        if eval_dict_copy.get(p) is None:
            if verbose: print("Entry "+str(p)+" NOT FOUND in EVAL")
            truth_dict_copy.pop(p)
    for p in eval_dict:
        if truth_dict_copy.get(p) is None:
            if verbose: print("Entry "+str(p)+" NOT FOUND in GROUND TRUTH")
            eval_dict_copy.pop(p)
    
    # Calculate coverage (% of truth_dict points remaining in the copy)
    if len(truth_dict) != 0:
        coverage = (float(len(truth_dict_copy)) / len(truth_dict)) * 100
    else:
        print("Ref dict empty")
        coverage = 100

    coverage = round(coverage,1)
    if verbose: print("Coverage Found: ",coverage)
    return truth_dict_copy, eval_dict_copy, coverage

# NEW VERSION, NO IMAGE, RETURNS DICT OF POINT ERRORS
def calc_error(truth_dict,eval_dict,precision=4,verbose=True):
    '''
    RETURNS [POINT ERROR DICT, SCALE FACTORS DICT, GLOBAL ERROR, GLOBAL ERROR STDDEV]
    '''
    # TODO: Implement draw_debug_image parameter that returns an image to 
    #       represent the calculation
    if verbose: print("CALCULATING ERROR")
    dict1 = truth_dict.copy()
    dict2 = eval_dict.copy()

    point_error_list = []
    point_errors = dict()
    dist_factors = dict()

    if verbose: print("POINT\tERROR")
    # For each reference point...
    for p in sorted(dict1.keys()):
        p1 = [dict1[p][0], dict1[p][1]]
        p2 = [dict2[p][0], dict2[p][1]]
        d1 = []
        d2 = []
        dist_factors[p] = dict()
        # Get distance to each other reference point...
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1]]
            p4 =[dict2[j][0], dict2[j][1]]
            if p1 == p3: continue
            gt_dist = pm.calculateDistance(p1[0],p1[1],p3[0],p3[1],precision)
            eval_dist = pm.calculateDistance(p2[0],p2[1],p4[0],p4[1],precision)
            scale_factor = gt_dist / eval_dist
            # if verbose:
            #     print("{}->{}: GT:{}   Eval:{}   Scale:{}".format(p,j,gt_dist,eval_dist,scale_factor))
            dist_factors[p][j] = scale_factor
            d1.append(gt_dist)
            d2.append(eval_dist)

        # Subtract the distances to get an error for each other reference 
        e = np.asarray(d1) - np.asarray(d2)
        e = np.abs(e)

        # Then get an average error for that reference point
        e = round(float(np.average(e)),precision)
        if verbose: print("{}\t{}".format(p,e))
        point_errors[p] = e
        point_error_list.append(e)

    # Average the average error for each point for a final score
    px_error = round(np.average(np.asarray(point_error_list)),precision)
    px_std = round(np.std(point_error_list),precision)

    if verbose: print("Global Error(px): ",px_error)
    if verbose: print("Std Dev(px): ",px_std,"\n")

    return [point_errors, dist_factors, px_error, px_std]


def calc_error_3d(ref_dict,test_dict,label="Not Specified",verbose=True):
    if verbose: print("CALCULATING ERROR")
    dict1 = ref_dict.copy()
    dict2 = test_dict.copy()

    largest_distance = 0
    largest_dist_string = "Unknown"
    dlist1 = []
    dlist2 = []
    errors = []
    error_avg = []

    # For each reference point...
    for p in sorted(dict1.keys()):
        print("\nGetting distances wrt ",p,"...")
        p1 = [dict1[p][0], dict1[p][1], dict1[p][2]]
        p2 = [dict2[p][0], dict2[p][1], dict2[p][2]]
        d1 = []
        d2 = []
        check_str = ""
        print("REF ",p,": ",p1)
        print("EVL ",p,": ",p2)
        # Get distance to each other reference point...
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1], dict1[j][2]]
            p4 =[dict2[j][0], dict2[j][1], dict2[j][2]]
            if p1 == p3: continue
            check_str = check_str + j + " "
            # Compute distances
            dist1 = np.linalg.norm(np.asarray(p1) - np.asarray(p3))
            if dist1 > largest_distance:
                largest_distance = dist1
                largest_dist_string = "(" + str(p) + " to " + str(j) + ")"
            d1.append(dist1)
            d2.append(np.linalg.norm(np.asarray(p2) - np.asarray(p4)))

        # Subtract the distances to get an error for each other reference 
        print("Checked in order: ", check_str)
        #print("D1\n",d1)
        #print("D2\n",d2)
        e = np.asarray(d1) - np.asarray(d2)
        e = np.around(e,decimals=2)
        
        e = np.abs(e)
        print("Distance Errors: ",e)
        # if verbose:  print(e)
        errors.append(list(e))
        # Then get an average error for that reference point
        e = round(float(np.average(e)),2)
        error_avg.append(e)
        dlist1.append(d1)
        dlist2.append(d2)

    if verbose: print("\nLargest Groundtruth Distance (For Scale): ",round(largest_distance,2), largest_dist_string)
    if verbose: print("Error Averages: ",error_avg)
    # Average the average error for each point for a final score
    px_error = round(np.average(np.asarray(error_avg)),2)
    px_std = round(np.std(error_avg),2)

    if verbose: print("Error Metric: ",px_error)
    if verbose: print("Std dev: ",px_std,"\n")

    return px_error, px_std

def auto_fullprocess(ref_dict,test_dict,test_img,v=True,visualize=True):
    # original_ref_dict = ref_dict.copy()
    # original_test_dict = test_dict.copy()
    processed_img = test_img.copy()
    draw_img = test_img.copy()
    draw_img.fill(255)
    print(v)
    #1. Remove entries that aren't present in both dicts, calculate coverage
    ref_dict, test_dict, coverage = calc_coverage(ref_dict, test_dict, verbose=v)
    
    #2. Get the midpoints for each dict and align test set to the reference
    aligned_dict, xy_off = auto_align(ref_dict, test_dict, draw_img)
    if visualize:
        cv.imshow("Map Comparison",draw_img)
        cv.waitKey(0)

    #3. Find the proper scale to minimize error metric

    scaled_dict, min_scale, scale_img = auto_scale(ref_dict,aligned_dict,draw_img, verbose=v)
    if visualize:
        cv.imshow("Map Comparison",scale_img)
        cv.waitKey(0)
    
    #4. Find proper rotation by minimizing the sum of distances between corresponding markers
    rot_dict, min_rot, rot_img = auto_rotate(ref_dict, scaled_dict, scale_img, verbose=v)
    if visualize:
        cv.imshow("Map Comparison", rot_img)
        cv.waitKey(0)

    # test_img = trans_image.image.copy()
    if v:
        print("FINAL TRANSFORM RESULTS")
        print("Coverage:{}  XY_Offset:{}  Scale:{}  Rotation:{}".format(coverage,xy_off,min_scale,min_rot))
    
    #5. Get transform matrix, apply it to test image
    marker_mp = get_dict_midpoint(test_dict)
    rot_mat = cv.getRotationMatrix2D(marker_mp, -min_rot, min_scale)
    rot_mat[0][2] = rot_mat[0][2] - xy_off[0]
    rot_mat[1][2] = rot_mat[1][2] - xy_off[1]
    rotated_img = cv.warpAffine(processed_img, rot_mat, processed_img.shape[1::-1], flags=cv.INTER_LINEAR,borderValue=(255,255,255))
    
    # if visualize:
    #     cv.imshow("Map Comparison", rotated_img)
    #     cv.waitKey()

    # Return tranformed image and marker dict
    return rotated_img, rot_dict

def generate_pointerror_contour_plot(truth_dict,eval_dict,metrics,image=None,save_file=None,title="Point Error Contour",units="px",note=""):
    scores_dict = metrics[0]
    flip_x = False
    flip_y = True
    draw_labels = True
    sparse_x = []
    sparse_y = []
    sparse_error = []
    for k in eval_dict:
        sparse_x.append(eval_dict[k][0])
        sparse_y.append(eval_dict[k][1])
        sparse_error.append(scores_dict[k])

    vmin = min(sparse_error)
    vmax = max(sparse_error)
    # vmin = 0 
    # vmax = 50

    # Create a regular grid for interpolation
    grid_resolution = 250
    pad = 50
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Interpolate sparse data onto the regular grid
    Z = griddata((sparse_x, sparse_y), sparse_error, (X, Y), method='linear')

    # Create a contour plot
    # Create a figure with a subplot grid (2 rows, 1 column)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # plt.figure(figsize=(16, 9))
    if image is not None:
        print("Adding image to contour plot")
        ax1.imshow(image,zorder=1)

    contour = ax1.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r',alpha=0.7,linewidths=0,vmin=vmin, vmax=vmax)
    # Draw Scatter Points
    ax1.scatter(sparse_x, sparse_y, c='black', cmap='rainbow_r', alpha=0.5, s=4)

    # Set Axis
    if flip_x:
        x_min = max(x)
        x_max = min(x)
    else:
        x_min = min(x)
        x_max = max(x)
    if flip_y:
        y_min = max(y)
        y_max = min(y)
    else:
        y_min = min(y)
        y_max = max(y)
    ax1.axis([x_min, x_max, y_min, y_max])
    ax2.axis([0, 1, 0, 1])
    colorbar = plt.colorbar(contour,label="Global Error ({})".format(units))
    # colorbar.set_clim(0, max(sparse_error))   

    # for x, y, label in zip(sparse_x, sparse_y, sparse_error):
    if draw_labels:
        for k in eval_dict:
            ax1.text(eval_dict[k][0], eval_dict[k][1], k, color='black', fontsize=8, ha='left', va='top')

    
    label_str = "Coverage: {}%\nGlobal Error: {}{}\nStd. Dev: {}{}\n{}".format(
        metrics[4],metrics[2],units,metrics[3],units,note
    )
    # Add text or annotations to the second subplot (ax2)
    ax2.text(0.01, 0.01, label_str, fontsize=12, color='black')
    # Remove axis labels and ticks for the second subplot
    ax2.axis('off')
    # Adjust the layout spacing
    # plt.tight_layout()

    ax1.set_title(title)
    if save_file is not None:
        # plt.show()
        plt.savefig(save_file)
        print("Saved: {}".format(save_file))
        plt.close()
    else:
        plt.show()
        
def generate_scalefactor_plot(truth_dict,eval_dict,metrics,excludestd=0,image=None,save_file=None,title="Scale Factor Plot",units="px",note=""):
    n = len(metrics[0])
    print(f"Using histogram with 5N ({n*5}) bins. (N={n})")
    precision = 7
    flip_x = False
    flip_y = True
    draw_labels = True

    # Coordinates for all point pairs and their corresponding scale value
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scale = []
    dist_dict = metrics[1]
    for p in dist_dict:
        for k in dist_dict[p]:
            x1.append(eval_dict[p][0])
            y1.append(eval_dict[p][1])
            x2.append(eval_dict[k][0])
            y2.append(eval_dict[k][1])
            scale.append(dist_dict[p][k])

    # Coordinates used for scatter plot
    sparse_x = []
    sparse_y = []
    for k in eval_dict:
        sparse_x.append(eval_dict[k][0])
        sparse_y.append(eval_dict[k][1])


    # Calculate stats on scale values
    vmin = min(scale)
    vmax = max(scale)
    print("Scale min:{}  max:{}".format(vmin,vmax))
    scale_avg = round(np.average(np.asarray(scale)),precision)
    scale_stddev = round(np.std(scale),precision)
    print("Avg:{}  StdDev:{}".format(scale_avg,scale_stddev))
    # Set vmin and vmax for the colorscale
    if scale_stddev < 0.0000001:
        vmin = scale_avg - 0.2
        vmax = scale_avg + 0.2
    else:
        vmin = scale_avg - (3 * scale_stddev)
        vmax = scale_avg + (3 * scale_stddev)
    # vmin = 0 
    # vmax = 50

    # Create a regular grid for interpolation
    grid_resolution = 250
    pad = abs((max(sparse_x)-min(sparse_x))*0.05)
    print("AX1 Padding: ",pad)
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)

    # Create a figure with a subplot grid (2 rows, 1 column)
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Add background image to figure
    if image is not None:
        print("Adding image to contour plot")
        ax1.imshow(image,zorder=1)

    # Create custom color map
    cmap1 = plt.get_cmap('RdYlGn')
    cmap2 = plt.get_cmap('RdYlGn_r')
    colors = []
    n_segments = 128
    for i in range(n_segments):
        t = i / (n_segments - 1)
        colors.append(cmap1(t))
    for i in range(n_segments):
        t = i / (n_segments - 1)
        colors.append(cmap2(t))
    custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn_RdYlGn_r', colors, N=n_segments*2)

    # colormap = mpl.colormaps['RdYlGn']
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    for i in range(len(x1)):
        if scale[i] <= scale_avg-(scale_stddev*excludestd) or scale[i] >= scale_avg+(scale_stddev*excludestd):
            color = sm.to_rgba(scale[i])  # Map the value to a color
            ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], color=color, alpha=0.5)

    # Plot Scale Factor Data
    hist, bins, _ = ax2.hist(scale, bins=5*n, edgecolor='black', alpha=1.0)
    max_value = max(hist)
    # Draw Exclusion Zone
    ax2.plot([scale_avg+(scale_stddev*excludestd),scale_avg+(scale_stddev*excludestd)],[0,max_value],color='black')
    ax2.plot([scale_avg-(scale_stddev*excludestd),scale_avg-(scale_stddev*excludestd)],[0,max_value],color='black')
    ax2.add_patch(Rectangle((scale_avg-(scale_stddev*excludestd),0),(scale_stddev*excludestd*2),max_value,facecolor = 'black',fill=True,alpha=0.3))
    # Draw Mean and Std Dev markers
    ax2.plot([scale_avg,scale_avg],[0,max_value],color='green')
    ax2.plot([scale_avg+scale_stddev,scale_avg+scale_stddev],[0,max_value],color='yellow')
    ax2.plot([scale_avg-scale_stddev,scale_avg-scale_stddev],[0,max_value],color='yellow')
    ax2.plot([scale_avg+(scale_stddev*2),scale_avg+(scale_stddev*2)],[0,max_value],color='orange')
    ax2.plot([scale_avg-(scale_stddev*2),scale_avg-(scale_stddev*2)],[0,max_value],color='orange')
    ax2.plot([scale_avg+(scale_stddev*3),scale_avg+(scale_stddev*3)],[0,max_value],color='red')
    ax2.plot([scale_avg-(scale_stddev*3),scale_avg-(scale_stddev*3)],[0,max_value],color='red')
    label_str = "Mean: {:f}\nStdDev: {:f}\nNormStdDev: {:f}".format(scale_avg,scale_stddev,scale_stddev/scale_avg)
    print(label_str)
    # ax2.text(bins[0], max_value, label_str, fontsize=10, color='black',va='top')
    # Draw Scatter Points
    ax1.scatter(sparse_x, sparse_y, c='black', alpha=0.5, s=4)

    # Set Axis
    if flip_x:
        x_min = max(x)
        x_max = min(x)
    else:
        x_min = min(x)
        x_max = max(x)
    if flip_y:
        y_min = max(y)
        y_max = min(y)
    else:
        y_min = min(y)
        y_max = max(y)
    ax1.axis([x_min, x_max, y_min, y_max])
    print("AX1 Axis: ",[x_min, x_max, y_min, y_max])
    # ax2.axis([0, vmax, 0, 20])
    # ax2.axis([0, 1, 0, 1])
    cbar = plt.colorbar(sm, label='Scale Values',ax=ax1)
    # colorbar.set_clim(0, max(sparse_error))   

    # for x, y, label in zip(sparse_x, sparse_y, sparse_error):
    if draw_labels:
        for k in eval_dict:
            ax1.text(eval_dict[k][0], eval_dict[k][1], k, color='black', fontsize=8, ha='left', va='top')

    # Add text or annotations to the second subplot (ax2)
    # label_str = "Coverage: {}%\nGlobal Error: {}{}\nStd. Dev: {}{}\n{}".format(
    #     metrics[4],metrics[2],units,metrics[3],units,note
    # )
    # ax2.text(0.01, 0.01, label_str, fontsize=12, color='black')
    # Remove axis labels and ticks for the second subplot
    # ax2.axis('off')
    # Adjust the layout spacing
    # plt.tight_layout()

    ax1.set_title(title)
    ax2.set_title("Connection Scale Factor Histogram")
    ax2.set_xlabel("Scale Factor")
    ax2.set_ylabel("Count")
    if save_file is not None:
        # plt.show()
        plt.savefig(save_file)
        print("Saved: {}".format(save_file))
        plt.close()
    else:
        plt.show()

    return [scale_avg,scale_stddev,scale_stddev/scale_avg]