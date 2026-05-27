import os
from os.path import join, dirname, basename, exists, abspath
import cv2 as cv
import json
import point_manip as pm
import csv_loader
import utilities as util
import calculate_metrics as metrics

def load_config(config_path=None,verbose=True):
    """
    Loads and validates configuration from a JSON file one directory above the script.
    """
    if config_path is None:
        script_dir = dirname(abspath(__file__))
        config_path = join(script_dir, "..", "reference_eval_config.json")

    print(f"Loading Config: {config_path}")

    if not exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if verbose:
        print("Configuration Loaded: ")
        print(json.dumps(config, indent=4))

    return config

def calc_available_points(gt_data, point_denominations,verbose=True):
    if verbose: print("Calculating Total Available Points:")
    available_points = 0
    for p in gt_data:
        found = False
        for k in point_denominations.keys():
            if k in p['label']:
                available_points += point_denominations[k]
                if verbose: print(f"{p}: +{point_denominations[k]}")
                found = True
                break
        if not found:
            print(f"WARNING: GT Point {p} has no defined point denomination!")
    
    print(f"Total Available Points: {available_points}")
    return available_points

def calc_point_error_results(point_error_dict, point_denominations, threshold=0.5):
    point_sum = 0
    discard_thresh = 2.5
    discard_list = []
    obj_type_tally = dict()

    for key in point_error_dict:
        points = -1
        obj_type = "NONE"
        for k in point_denominations.keys():
            if k in key:
                obj_type = k
        if obj_type == "NONE":
            print("WARNING: Point denomination for {key} not found, skipping...")
            continue

        points = point_denominations.get(obj_type,-1)
        val = point_error_dict[key]
        if val <= threshold:
            print(f"[{key}][{val:.3f}]\tSCORE +{points}")
            point_sum += points
            obj_type_tally[obj_type] = obj_type_tally.get(obj_type,0) + 1
        else:
            print(f"[{key}][{val:.3f}]\t--")

        if val >= discard_thresh:
            discard_list.append(key)
    print(f"\nSCORES:\nTYPE\t#\tSCORE")
    for o in obj_type_tally:
        print(f"{o}\t{obj_type_tally.get(o,0)}\t{point_denominations.get(o,0)*obj_type_tally.get(o,0)}")
    print(f"TOTAL DETECTION SCORE: {point_sum}\n")

    if len(discard_list) > 0:
        print(f"DISCARD LIST (Error > {discard_thresh}):")
        for d in discard_list:
            print(f"  > {d}: {point_error_dict[d]}")
        # print(point_error_dict)
    return point_sum, discard_list



if __name__ == "__main__":
    # LOAD CONFIG FILE ================
    config = load_config()
    # TODO: Implement command line argument overrides

    # ARGUMENTS FROM CONFIG
    verbose = config["verbose"]         # Bool: Print all the details
    log_results = config["log_results"] # Bool: Log metrics to csv
    log_name = config["log_file"]
    eval_3d = config["eval_3d"]         # Bool: Perform 3D or 2D evaluation
    event_name = config["event_name"]
    team_name = config["team_name"]
    test_note = config["note"]
    use_headers = config["inject_headers"] # List: Set to None/null if first line of csv's have headers

    # Establish Output Directory ===============================================
    if config["root_directory"] is None:
        output_dir = util.select_directory(".", "Select output dir")
    else:
        output_dir = config["root_directory"]
    assert output_dir != None, "Output Directory cannot be 'None'."
    if not os.path.exists(output_dir):
        input(f"Create output directory {output_dir}? (Ctrl+C to cancel)")
        os.makedirs(output_dir)
        print(f"Created Output Dir: {output_dir}")
    # Establish log file in output directory
    log_file = os.path.join(output_dir,log_name)
  
    # Establish Ground Truth File ===============================================
    if config["ground_truth_file"] is None:
        ground_truth_path = csv_loader.open_csv_dialog(output_dir)
        assert ground_truth_path != None, "You must select a ground truth file."
    else:
        ground_truth_path = join(output_dir, config["ground_truth_file"])
    gt_file = ground_truth_path

    # Load Ground Truth Points ==============================
    print("\nLoading Ground Truth Reference Points...")
    headers1, data_gt = csv_loader.read_csv_points(gt_file,use_headers,verbose=verbose)
    gt_filename = os.path.basename(gt_file)
    data_gt = csv_loader.fix_data_types(data_gt,set_str=['label'],set_float=['x','y','z'])


    # Establish Evaluation File ===============================================
    if config["eval_file"] is None:
        eval_pts_path = csv_loader.open_csv_dialog(output_dir)
        assert eval_pts_path != None, "You must select an evaluation file."
    else:
        eval_pts_path = join(output_dir, config["eval_file"])
    eval_file = eval_pts_path

    # Load Evaluation Points =============================
    print("\nLoading Evaluation Reference Points...")
    if not config["robocup_eval_format"]:
        headers2, data_eval = csv_loader.read_csv_points(eval_file,use_headers,verbose=verbose)
        eval_filename = os.path.basename(eval_file)
        data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])
    else:
        # MODIFIED FOR ROBOCUP FORMAT (Loads data correctly from specified Robocup csv format)
        print(" > Modifying eval reference points based on RoboCup csv format...")
        headers2, data_eval = csv_loader.read_csv_points(eval_file,headers = None,skip_rows=8,verbose=True)
        eval_filename = os.path.basename(eval_file)
        data_eval = csv_loader.append_concatenated_header(data_eval,'label',['type','name'],True)
        data_eval = csv_loader.extract_fields(data_eval,['label','x','y','z'])
        data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])
        detections_out = os.path.dirname(eval_file)
        csv_loader.write_csv(join(detections_out,"FIXED-"+eval_filename),['label','x','y','z'],data_eval)
    

    # OPTIONS FOR PLOT GENERATION (2D EVAL ONLY) ======================================
    plt_config = config["plot_generation"]
    create_plots = plt_config["create_plots"]
    save_plots = plt_config["save_plots"]
    show_plots = plt_config["show_plots"]
    figsize_inches = tuple(plt_config["fig_size"])
    auto_scale = plt_config["auto_scale"] # Whether to plot re-scaled metrics
    error_units = plt_config["units"] # Set to None for unspecified
    error_to_plot = 'scaled_point_errors' # or 'point_errors'
    exclude_std_devs = plt_config["sf_exclude_std_dev"]
    if save_plots:
        err_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"point-errors"))
        scale_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"scale-factors"))
    else:
        err_plot_name = None
        scale_plot_name = None

    # > Optional: Map Image Background for 2D plots 
    #   (reference points must be in pixel space, works well with map_labeller.py tool)
    if plt_config["include_background_image"]:
        # Select image file
        if plt_config["background_image"] is None:
            map_image_file = util.open_image_dialog(".")
        else:
            map_image_file = plt_config["background_image"]
        # Load image 
        map_image = cv.imread(map_image_file)
        map_image = cv.cvtColor(map_image,cv.COLOR_BGR2RGB)
    else:
        map_image = None


    # PAUSE BEFORE CALCULATION =======================================
    if config["pause_before_running"]:
        input("Continue?: Ctrl+C to exit")
    

    # CALCULATE METRICS ============================================
    print("\nCalculating Metrics...")
    metric = metrics.initialize_metrics()

    # 1. CALCULATE COVERAGE
    metric['cvg_total'] = len(data_gt) # Number of ground truth labels
    original_data_gt = data_gt.copy()
    data_gt, data_eval, metric['coverage'] = metrics.calc_coverage(data_gt,data_eval,verbose=verbose)
    metric['cvg_found'] = len(data_eval) # Number of matching points found in eval file


    # 2. CALCULATE GLOBAL ERROR METRICS
    metric['point_errors'], metric['error_avg'], metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    
    
    # 3. CALCULATE SCALE FACTOR METRICS
    metric['point_scales'],metric['scale_avg'],metric['scale_std'] = metrics.calc_scale_factors(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    metric['norm_scale_std'] = metric['scale_std'] / metric['scale_avg']


    # 4. RE-SCALE AND FIND ERROR METRICS AGAIN
    print(f"Scaling Eval Points based on found scale factor average ({metric['scale_avg']:.2f})...")
    scaled_data_eval = pm.scale_points_wrt_origin(data_eval,metric['scale_avg'],eval_3d)
    # > RE-CALCULATE GLOBAL ERROR METRICS
    metric['scaled_point_errors'],metric['scaled_error_avg'],metric['scaled_error_std'] = metrics.calc_error(data_gt, scaled_data_eval,use_z=eval_3d,verbose=verbose)
    # > RE-CALCULATE SCALE FACTOR METRICS
    metric['scaled_point_scales'], metric['scaled_scale_avg'], metric['scaled_scale_std'] = metrics.calc_scale_factors(data_gt, scaled_data_eval,use_z=eval_3d,verbose=verbose)
    metric['scaled_norm_scale_std'] = metric['scaled_scale_std'] / metric['scaled_scale_avg']


    # 5. CALCULATE POINTS FOR ROBOCUP MAPPING CHALLENGE:
    print("Calculating Scores For Robocup:")
    point_denoms = config["id_points_by_type"]
    print(f"Using Point Denominations:\n{point_denoms}")
    metric['available_points'] = calc_available_points(original_data_gt,point_denoms,verbose)
    metric['detection_thresh'] = config["scoring_threshold"]
    metric['detection_score'], discard_list = calc_point_error_results(metric['point_errors'],point_denoms,metric['detection_thresh'])
    # for k in discard_list:
    #     for x, y in zip(data_eval, data_gt):
    #         data_eval.pop(k)
    #         data_gt.pop(k)
    # metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    # metric['detection_score'], data_eval = print_point_error_results(metric['point_errors'],metric['detection_thresh'])

    # UNIT SCALING (Ignore this)
    metric_scaler = 1.0 #0.033311475409836 
    s = metric_scaler

    # GENERATE OUTPUT PLOTS (2D ONLY)
    if create_plots and not eval_3d:
        # CREATE POINT ERROR CONTOUR PLOT
        error_to_plot = "scaled_point_error" if auto_scale else "point_error"
        metrics.generate_pointerror_contour_plot(data_eval,metric[error_to_plot],metric,map_image,err_plot_name,show_plots,figsize_inches,error_units,s)
        # CREATE SCALE FACTOR PLOT
        if auto_scale:
            if map_image is not None:
                scaled_map_image = util.scale_image_with_aspect_ratio(map_image,metric['scale_avg'])
            # data_eval = scaled_data_eval
            metrics.generate_scalefactor_plot(scaled_data_eval,metric,exclude_std_devs,scaled_map_image,scale_plot_name,show_plots,figsize_inches,True)
        else:
            metrics.generate_scalefactor_plot(data_eval,metric,exclude_std_devs,map_image,scale_plot_name,show_plots,figsize_inches,False)
    
    # CALCULATE MAPPING SCORE (ROBOCUP)
    metric["raw_mapping_score"] = metric['detection_score'] / (1 + metric['error_avg']*s)
    print(f"Detection Points (P): {metric['detection_score']} / {metric['available_points']} = {metric['detection_score']/metric['available_points']:.2f}")
    print(f"Avg Error (E): {metric['error_avg']*s:.2f}")
    print(f"Raw ID Score (P/(1+E)): {metric['raw_mapping_score']:.2f}")
    metric["mapping_score"] =  (metric["raw_mapping_score"] * 100) / metric['available_points']
    print(f"Adjusted ID Score: {metric['mapping_score']:.2f}")


    # LOG RESULTS
    if log_results:
        # Write metrics results to specified CSV file.
        if eval_3d:
            test_note = test_note + " (3D Evaluation)"

        log_list = [util.timestamp(),event_name, team_name, test_note,
                    gt_filename,eval_filename,
                    metric['cvg_found'],metric['cvg_total'],metric['coverage'],
                    metric['error_avg']*s,metric['error_std']*s,
                    metric['scale_avg'],metric['scale_std'],metric['norm_scale_std'],
                    metric['scaled_error_avg']*s,metric['scaled_error_std']*s,
                    metric['detection_score'],metric["available_points"],metric['detection_thresh'],
                    metric["raw_mapping_score"],metric["mapping_score"]]
        metrics.log_to_csv(log_file, log_list,verbose=verbose)
    
        