import os
import csv
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
import point_manip as pm
import utilities as util

def log_to_csv(log_file, data_values,header=None,verbose=False,unit="(m)"):
    if verbose: print(f"Logging Metrics to: {log_file}")
    if header is None:
        header = ['Timestamp', 'Map Name', 'Note',
             'Ground Truth File','Evaluation File',
             'Coverage Found','Coverage Total','Coverage %', 
             f'Error Average {unit}', f'Error Std Dev {unit}',
             'Scale Average', 'Scale Std Dev', 'Normalized Scale Std Dev',
             'Scaled Error Average', 'Scaled Error Std Dev',
             'Detection Score','Detection Thresh','Mapping Score']  # Adjust the header columns as needed
    
    # Check if the directory exists, create it if it does not
    log_dir = os.path.dirname(log_file)
    log_name = os.path.basename(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_exists = os.path.isfile(log_file)
    
    # Open the file in append mode
    with open(log_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # If the file does not exist, write the header first
        if not file_exists:
            print("LOG CREATED...")
            writer.writerow(header)
        
        # Append the data values
        writer.writerow(data_values)
        if not verbose:
            print("LOG DATA ADDED...")
        else:
            print(f"LOG DATA ADDED:\n{data_values}")

def initialize_metrics(metric_list=None):
    if metric_list is None:
        metric_list = ['cvg_found','cvg_total','coverage',
                       'gt_file','eval_file',
                       'point_errors','error_avg','error_std',
                       'point_scales','scale_avg','scale_std','norm_scale_std',
                       'scaled_point_errors','scaled_error_avg','scaled_error_std']
    metric = dict()
    for m in metric_list:
        metric[m] = None
    return metric

def calc_coverage(data_gt,data_eval,match_by='label',verbose=True):
    '''
    RETURNS [Matching Truth Dict, Matching Eval Dict, Coverage Score]
    '''
    if verbose: print("CALCULATING COVERAGE")
    ground_truth_copy = data_gt.copy()
    eval_copy = data_eval.copy()
    
    matched_gt, matched_eval = pm.match_data_by_field(ground_truth_copy,eval_copy,match_by,verbose=verbose)

    # Calculate coverage (% of truth_dict points remaining in the copy)
    if len(ground_truth_copy) != 0:
        coverage = (float(len(matched_gt)) / len(ground_truth_copy)) * 100
    else:
        print("Ground truth data empty... Full coverage.")
        coverage = 100

    coverage = round(coverage,1)
    print(" > COVERAGE METRICS")
    print(f"Coverage Found: {len(matched_gt)}/{len(ground_truth_copy)} -> {coverage}\n")
    return matched_gt, matched_eval, coverage

def calc_error(data_gt,data_eval,use_z=False,match_by='label',verbose=True):
    '''
    RETURNS [POINT ERROR DICT, GLOBAL ERROR, GLOBAL ERROR STDDEV, SCALE FACTORS DICT]
    '''
    # TODO: Implement draw_debug_image parameter that returns an image to 
    #       represent the calculation
    if verbose: print(f"\nCALCULATING ERROR {'(3D)'if use_z else ''}")
    # Verify entries between datasets are matching
    data_gt, data_eval = pm.match_data_by_field(data_gt,data_eval,match_by,verbose=False)
    # Precompute lookup dictionaries to easily find data based on the match_by (points label) field
    gt_lookup_dict = {entry[match_by]: entry for entry in data_gt}
    eval_lookup_dict = {entry[match_by]: entry for entry in data_eval}

    point_error_list = []
    point_errors = dict()

    # For each reference point...
    for p in data_gt:
        gt_lbl = p[match_by]
        if not use_z:
            gt_p1 = [p['x'],p['y']]
            ev_p1 = [eval_lookup_dict[gt_lbl]['x'],eval_lookup_dict[gt_lbl]['y']]
        else:
            gt_p1 = [p['x'],p['y'],p['z']]
            ev_p1 = [eval_lookup_dict[gt_lbl]['x'],eval_lookup_dict[gt_lbl]['y'],eval_lookup_dict[gt_lbl]['z']]
        gt_dists = []
        ev_dists = []
        if verbose: print(f" > LABEL: {gt_lbl}")
        # Get distance to each other reference point...
        for j in data_eval:
            ev_lbl = j[match_by]
            if ev_lbl == gt_lbl: continue
            if not use_z:
                gt_p2 = [gt_lookup_dict[ev_lbl]['x'],gt_lookup_dict[ev_lbl]['y']]
                ev_p2 = [j['x'],j['y']]
            else:
                gt_p2 = [gt_lookup_dict[ev_lbl]['x'],gt_lookup_dict[ev_lbl]['y'],gt_lookup_dict[ev_lbl]['z']]
                ev_p2 = [j['x'],j['y'],j['z']]
            gt_dist = pm.calc_distance(gt_p1,gt_p2)
            ev_dist = pm.calc_distance(ev_p1,ev_p2)
            abs_diff = np.abs(gt_dist - ev_dist)
            if verbose:
                print(f"{gt_lbl}->{ev_lbl}: GT:{gt_dist:.2f}  EV:{ev_dist:.2f}  AbsDiff:{abs_diff:.2f}")
            # Store calculations
            gt_dists.append(gt_dist)
            ev_dists.append(ev_dist)
        # Subtract the distances to get an error for each other reference 
        e = np.asarray(gt_dists) - np.asarray(ev_dists)
        e = np.abs(e)
        # Then get an average error for that reference point
        e = float(np.average(e))
        if verbose: print(f"Average Error: {e:.2f}")
        point_errors[gt_lbl] = e
        point_error_list.append(e)

    # Average the average error for each point for a final score
    error_avg = np.average(np.asarray(point_error_list))
    error_std = np.std(point_error_list)
    
    print(f" > ERROR METRICS {'(3D)'if use_z else ''}")
    print(f"Average Global Error: {error_avg:.4f}")
    print(f"Std Dev: {error_std:.4f}\n")

    return [point_errors, error_avg, error_std]

def calc_scale_factors(data_gt,data_eval,use_z=False,match_by='label',verbose=True):
    '''
    RETURNS [SCALE FACTOR DICT, SCALE FACTOR AVG, SCALE FACTOR STDDEV]
    '''
    # TODO: Implement draw_debug_image parameter that returns an image to 
    #       represent the calculation
    if verbose: print(f"\nCALCULATING SCALE FACTORS {'(3D)'if use_z else ''}")
    # Verify entries between datasets are matching
    data_gt, data_eval = pm.match_data_by_field(data_gt,data_eval,match_by,verbose=False)
    # Precompute lookup dictionaries to easily find data based on the match_by (points label) field
    gt_lookup_dict = {entry[match_by]: entry for entry in data_gt}
    eval_lookup_dict = {entry[match_by]: entry for entry in data_eval}
    scale_factor_list = []
    scale_factors = dict()

    # For each reference point...
    for p in data_gt:
        gt_lbl = p[match_by]
        if not use_z:
            gt_p1 = [p['x'],p['y']]
            ev_p1 = [eval_lookup_dict[gt_lbl]['x'],eval_lookup_dict[gt_lbl]['y']]
        else:
            gt_p1 = [p['x'],p['y'],p['z']]
            ev_p1 = [eval_lookup_dict[gt_lbl]['x'],eval_lookup_dict[gt_lbl]['y'],eval_lookup_dict[gt_lbl]['z']]
        point_scales = []
        scale_factors[gt_lbl] = dict()
        if verbose: print(f" > LABEL: {gt_lbl}")
        # Get distance to each other reference point...
        for j in data_eval:
            ev_lbl = j[match_by]
            if ev_lbl == gt_lbl: continue
            if not use_z:
                gt_p2 = [gt_lookup_dict[ev_lbl]['x'],gt_lookup_dict[ev_lbl]['y']]
                ev_p2 = [j['x'],j['y']]
            else:
                gt_p2 = [gt_lookup_dict[ev_lbl]['x'],gt_lookup_dict[ev_lbl]['y'],gt_lookup_dict[ev_lbl]['z']]
                ev_p2 = [j['x'],j['y'],j['z']]
            gt_dist = pm.calc_distance(gt_p1,gt_p2)
            ev_dist = pm.calc_distance(ev_p1,ev_p2)
            scale_factor = gt_dist / ev_dist
            scale_factors[gt_lbl][ev_lbl] = scale_factor
            scale_factor_list.append(scale_factor)
            if verbose:
                print(f"{gt_lbl}->{ev_lbl}: GT:{gt_dist:.2f}  EV:{ev_dist:.2f}  Scale:{scale_factor:.2f}")
            # Store calculation
            point_scales.append(scale_factor)
        # Then get an average scale factor for that reference point
        s = float(np.average(point_scales))
        if verbose: print(f"Average Scale Factor: {s:.2f}")
        # scale_factors[gt_lbl] = s
        # scale_factor_list.append(s)

    # Average the average scale for each point for a final score
    scale_avg = np.average(np.asarray(scale_factor_list))
    scale_std = np.std(scale_factor_list)
  
    print(f" > SCALE FACTOR METRICS {'(3D)'if use_z else ''}")
    print(f"Average Scale Factor: {scale_avg:.4f}")
    print(f"Std Dev: {scale_std:.4f}\n")

    return [scale_factors, scale_avg, scale_std]


def generate_pointerror_contour_plot(data,point_errors,metrics,image=None,save_file=None,show_plot=True,figsize=(8,8),unit=None,unit_scale=1.0):
    flip_x = False
    flip_y = True
    draw_labels = True
    sparse_x = []
    sparse_y = []
    sparse_error = []
    for p in data:
        sparse_x.append(p['x'])
        sparse_y.append(p['y'])
        sparse_error.append(point_errors[p['label']]*unit_scale)

    # Min and Max error values for color scaling
    vmin = min(sparse_error)
    vmax = max(sparse_error)
    vmin = 0 
    vmax = 3

    # Create a regular grid for interpolation
    grid_resolution = 500
    pad = 50
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Interpolate sparse data onto the regular grid
    Z = griddata((sparse_x, sparse_y), sparse_error, (X, Y), method='linear')

    # Create the figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Draw background map image
    if image is not None:
        print("Adding image to contour plot")
        ax1.imshow(image,zorder=1)

    # Draw Contour Plot
    # Constrain Z values
    if vmax < max(sparse_error):
        print(f"Constraining Z [{vmin}-{vmax}]")
        Z = np.clip(Z, vmin, vmax)
    # specify levels from vmim to vmax
    step = (vmax-vmin)/50
    levels = np.arange(vmin, vmax+step, step)
    contour = ax1.contourf(X, Y, Z, levels=levels, cmap='RdYlGn_r',alpha=0.7,vmin=vmin, vmax=vmax)
    color_ticks = np.linspace(vmin,vmax,11)
    # print(color_ticks)
    color_label = f"Global Error ({unit})" if unit is not None else "Global Error"
    colorbar = plt.colorbar(contour,label=color_label,ticks=color_ticks)

    # Draw reference point labels
    if draw_labels:
        for k in data:
            ax1.text(k['x'], k['y'], k['label'], color='black', fontsize=8, ha='left', va='top',
                     bbox=dict(facecolor="white", edgecolor="none", pad=0,alpha=0.5))
            
    # Draw Reference Points
    ax1.scatter(sparse_x, sparse_y, c='black', alpha=0.75, s=4, edgecolors='white', linewidths=0.5)

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
    ax2.axis([0, 1, 1, 0])

    # Create lower subplot (metrics label)
    if unit is None: unit = ""
    s = unit_scale
    label_str = f"[TITLE] {metrics['test_name']} - {metrics['test_note']}\n" \
                + f"[GROUND TRUTH FILE] {metrics['gt_file']}\n" \
                + f"[EVALUATION FILE] {metrics['eval_file']}\n\n" \
                + f"[METRICS]\n" \
                + f"Coverage: {metrics['coverage']}% ({metrics['cvg_found']}/{metrics['cvg_total']} points)\n" \
                + f"Error Avg: {metrics['error_avg']*s:.2f}{unit} ({metrics['error_std']*s:.2f} std dev)\n" \
                + f"Scale Avg: {metrics['scale_avg']:.2f} ({metrics['scale_std']:.2f} std dev)\n" \
                + f"Scaled Error Avg: {metrics['scaled_error_avg']*s:.2f}{unit} ({metrics['scaled_error_std']*s:.2f} std dev)" 
    # Add text or annotations to the second subplot (ax2)
    ax2.text(0.01, 0.04, label_str, verticalalignment='top', fontsize=10, color='black')
    # Set the edge color and line width for the subplot's border (black border)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)
    # Hide the tick marks
    ax2.tick_params(axis='both', length=0)
    # Hide the tick labels
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # Adjust the layout spacing
    plt.tight_layout()

    # SAVE / SHOW PLOT ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    ax1.set_title(metrics['test_name'] + " (Point Errors)")
    if save_file is not None:
        plt.savefig(save_file,dpi=300)
        print("Saved: {}".format(save_file))
        # plt.close()
    if show_plot:
        plt.show()
        
def generate_scalefactor_plot(data,metrics,excludestd=0,image=None,save_file=None,show_plot=True,figsize=(8,8),use_scaled_data=True):
    if use_scaled_data:
        scales_key = 'scaled_point_scales'
        avg_key = 'scaled_scale_avg'
        std_key = 'scaled_scale_std'
        norm_key = 'scaled_norm_scale_std'
    else:
        scales_key = 'point_scales'
        avg_key = 'scale_avg'
        std_key = 'scale_std'
        norm_key = 'norm_scale_std'

    n = len(metrics[scales_key])
    print(f"Using histogram with 5N ({n*5}) bins. (N={n})")
    flip_x = False
    flip_y = True
    draw_labels = True

    # Create a lookup dictionary by the label field.
    data_lookup = {entry['label']: entry for entry in data}

    # Coordinates for all point pairs and their corresponding scale value
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scale = []
    for p in metrics[scales_key]:
        for k in metrics[scales_key][p]:
            x1.append(data_lookup[p]['x'])
            y1.append(data_lookup[p]['y'])
            x2.append(data_lookup[k]['x'])
            y2.append(data_lookup[k]['y'])
            scale.append(metrics[scales_key][p][k])

    # Coordinates used for scatter plot
    sparse_x = []
    sparse_y = []
    for k in data:
        sparse_x.append(k['x'])
        sparse_y.append(k['y'])

    # Calculate stats on scale values
    vmin = min(scale)
    vmax = max(scale)
    print("Scale min:{}  max:{}".format(vmin,vmax))
    scale_avg = metrics[avg_key]
    scale_stddev = metrics[std_key]
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
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)

    # Create a figure with a subplot grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.141)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Add background image to figure
    if image is not None:
        print("Adding image to plot")
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
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    # Draw connections that are not excluded with the appropriate color
    for i in range(len(x1)):
        if scale[i] <= scale_avg-(scale_stddev*excludestd) or scale[i] >= scale_avg+(scale_stddev*excludestd):
            color = sm.to_rgba(scale[i])  # Map the value to a color
            ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], color=color, alpha=0.5)

    
    # Draw reference point labels
    if draw_labels:
        for k in data:
            ax1.text(k['x'], k['y'], k['label'], color='black', fontsize=8, ha='left', va='top',
                     bbox=dict(facecolor="white", edgecolor="none", pad=0,alpha=0.5))
            
    # Draw Reference Points
    ax1.scatter(sparse_x, sparse_y, c='black', alpha=0.75, s=4, edgecolors='white', linewidths=0.5)


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
    # print("AX1 Axis: ",[x_min, x_max, y_min, y_max])
    cbar = plt.colorbar(sm, label='Scale Values',ax=ax1)

    # Plot Scale Factor Data Histogram
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
    
    # Add text or annotations to the second subplot (ax2)
    label_str = f"Mean: {metrics[avg_key]:.3f}\nStdDev: {metrics[std_key]:.3f}\n" \
                f"Normed: {metrics[norm_key]:.3f}"
    _, y_max = ax2.get_ylim()
    x_min, x_max = ax2.get_xlim()
    x_range = x_max - x_min
    ax2.text(x_min+(x_range/100), y_max-(y_max/100), label_str, verticalalignment='top', fontsize=10, color='white',
             path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])
    ax2.text(x_min+(x_range/100), y_max-(y_max/100), label_str, verticalalignment='top', fontsize=10, color='black')

    # Adjust the layout spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    # SET AXIS TITLES AND LABELS ===== ===== ===== ===== ===== ===== ===== ===== 
    ax1.set_title(metrics['test_name'] + f" (Scale Factors)\n[Hiding +/- {excludestd:.1f} dev]",pad=0)
    ax2.set_title("Connection Scale Factor Histogram")
    ax2.set_xlabel("Scale Factor", labelpad=0)
    ax2.set_ylabel("Count", labelpad=0)

    # SAVE / SHOW PLOT ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    if save_file is not None:
        plt.savefig(save_file,dpi=300)
        print("Saved: {}".format(save_file))
    if show_plot:
        plt.show()