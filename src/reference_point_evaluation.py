import os
from os.path import join, dirname, basename, exists, abspath
import cv2 as cv
import argparse
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


def print_point_error_results(point_error_dict, threshold=0.5):
    point_sum = 0
    discard_thresh = 2.5
    discard_list = []
    obj_type_tally = dict()
    points_dict = {
        "ar_tag": 1,
        "hazmat": 2,
        "real_object": 10,
        "heat_signature": 15
    }
    for key in point_error_dict:
        points = -1
        obj_type = "NONE"
        if "ar_tag" in key: 
            obj_type = "ar_tag"
            # points = points_dict[obj_type]
        if "hazmat" in key: 
            # points = 2
            obj_type = "hazmat"
        if "real_object" in key: 
            # points = 10
            obj_type = "real_object"
        if "heat_signature" in key: 
            # points = 15
            obj_type = "heat_signature"
        points = points_dict.get(obj_type,-1)
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
        print(f"{o}\t{obj_type_tally.get(o,0)}\t{points_dict.get(o,0)*obj_type_tally.get(o,0)}")
    print(f"TOTAL DETECTION SCORE: {point_sum}\n")

    print(f"SHOULD REMOVE: {discard_list}")
    print(point_error_dict)
    return point_sum, discard_list

def main(args):
    data_gt = args.data_gt
    data_eval = args.data_eval
    # Do points need to be scaled? 
    # CALCULATE METRICS ============================================
    metric = metrics.initialize_metrics()
    metric['test_name'] = args.test_name
    metric['test_note'] = args.test_note
    metric['gt_file'] = args.gt_file
    metric['eval_file'] = args.eval_file
    # CALCULATE COVERAGE
    metric['cvg_total'] = len(data_gt)
    data_gt, data_eval, metric['coverage'] = metrics.calc_coverage(data_gt,data_eval,verbose=args.verbose)
    # metric['cvg_found'] = int(metric['cvg_total'] * metric['coverage'] / 100)
    metric['cvg_found'] = len(data_eval)
    # CALCULATE GLOBAL ERROR METRICS
    metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    
    # FOR ROBOCUP GERMAN OPEN:
    metric['detection_thresh'] = args.scoring_threshold
    metric['detection_score'], discard_list = print_point_error_results(metric['point_errors'],metric['detection_thresh'])
    # for k in discard_list:
    #     for x, y in zip(data_eval, data_gt):
    #         data_eval.pop(k)
    #         data_gt.pop(k)
    # metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    # metric['detection_score'], data_eval = print_point_error_results(metric['point_errors'],metric['detection_thresh'])

    # CALCULATE SCALE FACTOR METRICS
    metric['point_scales'],metric['scale_avg'],metric['scale_std'] = metrics.calc_scale_factors(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    metric['norm_scale_std'] = metric['scale_std'] / metric['scale_avg']

    if auto_scale or True:
        # AUTO SCALE AND FIND ERROR METRICS AGAIN
        print(f"Scaling Eval Points based on found scale factor average ({metric['scale_avg']})...")
        scaled_data_eval = pm.scale_points_wrt_origin(data_eval,metric['scale_avg'],args.eval_3d)
        # CALCULATE GLOBAL ERROR METRICS
        metric['scaled_point_errors'],metric['scaled_error_avg'],metric['scaled_error_std'] = metrics.calc_error(data_gt, scaled_data_eval,use_z=args.eval_3d,verbose=args.verbose)
        # CALCULATE SCALE FACTOR METRICS
        if args.map_image is not None:
            scaled_map_image = util.scale_image_with_aspect_ratio(args.map_image,metric['scale_avg'])
        metric['scaled_point_scales'],metric['scaled_scale_avg'],metric['scaled_scale_std'] = metrics.calc_scale_factors(data_gt, scaled_data_eval,use_z=args.eval_3d,verbose=args.verbose)
        metric['scaled_norm_scale_std'] = metric['scaled_scale_std'] / metric['scaled_scale_avg']

    # UNIT SCALING
    metric_scaler = 1 #0.033311475409836 #1
    s = metric_scaler

    # GENERATE OUTPUT PLOTS
    if args.create_plots and not args.eval_3d:
        if args.create_err_plot:
            metrics.generate_pointerror_contour_plot(data_eval,metric[args.error_to_plot],metric,args.map_image,args.err_plot_name,args.show_plots,args.figsize,args.units,s)
        if args.create_scale_plot:
            if auto_scale:
                # data_eval = scaled_data_eval
                metrics.generate_scalefactor_plot(scaled_data_eval,metric,args.exclude_std_devs,scaled_map_image,args.scale_plot_name,args.show_plots,args.figsize,True)
            else:
                metrics.generate_scalefactor_plot(data_eval,metric,args.exclude_std_devs,args.map_image,args.scale_plot_name,args.show_plots,args.figsize,False)
    
    # CALCULATE MAPPING SCORE (ROBOCUP)
    metric["raw_mapping_score"] = metric['detection_score'] / (1 + metric['error_avg']*s)
    print(f"Detection Points (P): {metric['detection_score']} / {args.total_points} = {metric['detection_score']/args.total_points:.2f}")
    print(f"Avg Error (E): {metric['error_avg']*s:.2f}")
    print(f"Raw ID Score (P/(1+E)): {metric['raw_mapping_score']:.2f}")
    metric["mapping_score"] =  (metric["raw_mapping_score"] * 100) / args.total_points
    print(f"Adjusted ID Score: {metric['mapping_score']:.2f}")


    # LOG RESULTS
    if log_results:
        # Write metrics results to specified CSV file.
        if eval_3d:
            args.test_note = args.test_note + " (3D Evaluation)"
        log_list = [util.timestamp(),args.test_name,args.test_note,
                    metric['gt_file'],metric['eval_file'],
                    metric['cvg_found'],metric['cvg_total'],metric['coverage'],
                    metric['error_avg']*s,metric['error_std']*s,
                    metric['scale_avg'],metric['scale_std'],metric['norm_scale_std'],
                    metric['scaled_error_avg']*s,metric['scaled_error_std']*s,
                    metric['detection_score'],metric['detection_thresh'],metric["raw_mapping_score"],metric["mapping_score"]]
        metrics.log_to_csv(args.log_file, log_list,verbose=args.verbose)


if __name__ == "__main__":
    # LOAD CONFIG FILE ================
    # NOT YET IMPLEMENTED
    config = load_config()
    
    # DEFAULT ARGUMENTS 
    verbose = config["verbose"]
    log_results = config["log_results"]
    eval_3d = config["eval_3d"]
    auto_scale = config["auto_scale"]

    # output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/point_method"
    # default_output = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open/"
    output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open/"
    output_dir = config["root_directory"]
    # output_dir = util.select_directory(default_output, "Select output dir")
    log_name = config["log_file"]

    test_name = config["event_name"]
    test_note = config["note"]
    # Structural Reference Points  -  C Targets  -  All Points

    # Establish Ground Truth File
    if config["ground_truth_file"] is None:
        ground_truth_path = csv_loader.open_csv_dialog(output_dir)
    else:
        ground_truth_path = join(output_dir, config["ground_truth_file"])
    # d = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open" # Default dir
    # gt_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open/groundtruth/gt1_combined_v2.csv"
    # pts1_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test3.csv"
    # pts1_file = csv_loader.open_csv_dialog(d)
    pts1_file = ground_truth_path

    # Establish Evaluation File and outliers output file
    if config["eval_file"] is None:
        eval_pts_path = csv_loader.open_csv_dialog(output_dir)
    else:
        eval_pts_path = join(output_dir, config["eval_file"])
    # d = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open"
    # pts2_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/german-open/groundtruth/gt1_ar_tags.csv"
    # pts2_file = pts1_file
    # pts2_file = csv_loader.open_csv_dialog(d)
    pts2_file = eval_pts_path

    use_headers = None    # Set to None if first line of csv has headers
    use_headers = ['label','x','y','z']
    # if eval_3d: use_headers.append('z')   
    
    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens"
    # map_image_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/parrot-farsight-photo.png"
    # map_image_file = util.open_image_dialog(d)
    map_image_file = None
    
    create_plots = True
    figsize_inches = (9,9) # (8,8) default
    show_plots = False
    create_err_plot = True
    create_scale_plot = True# True
    error_to_plot = 'scaled_point_errors' # or 'scaled_point_errors'
    error_units = "m" # Set to None for unspecified
    # error_to_plot = 'scaled_point_errors' # or 'scaled_point_errors'
    exclude_std_devs = 0
    err_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"point-errors"))
    # err_plot_name = None
    scale_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"scale-factors"))
    # scale_plot_name = None

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Tool for calculating map accuracy metrics and creating plots.")
    parser.add_argument('-1', '--pts1_file', dest='pts1_file', type=str, default=pts1_file,
                        help='Provide path to ground truth points or set to \'choose\' to be prompted to select.')
    parser.add_argument('-2', '--pts2_file', dest='pts2_file', type=str, default=pts2_file,
                        help='Provide path to eval points or set to \'choose\' to be prompted to select.')
    # Output
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default=output_dir,
                        help='Directory for saving output log and plots')
    parser.add_argument('-l', '--log_name', dest='log_name', type=str, default=log_name,
                        help='Name for log file (include .csv)')
    parser.add_argument('-t', '--test_name', dest='test_name', type=str, default=test_name,
                        help='Primary label for test')
    parser.add_argument('-n', '--note', dest='test_note', type=str, default=test_note,
                        help='Secondary label for test')
    # Plotting
    parser.add_argument('-p', '--plot', dest='create_plots', type=bool, default=create_plots,
                        help='Whether to create metric plots')
    parser.add_argument('-e', '--exclude', dest='exclude_std_devs', type=int, default=exclude_std_devs,
                        help='Standard Deviations to exclude from drawing on scale plot')
    parser.add_argument('-m', '--map_image', dest='map_image_file', type=str, default=map_image_file,
                        help='Provide path or set to \'choose\' to be prompted to select.')
    # Global Bools
    parser.add_argument('-z', '--use-z', dest='eval_3d', type=bool, default=eval_3d,
                        help='Whether to perform 3D point evaluation.')
    parser.add_argument('-s', '--auto_scale', dest='auto_scale', type=bool, default=auto_scale,
                        help='Whether to auto-scale and do another error calculation.')
    parser.add_argument('-x', '--log_results', dest='log_results', type=bool, default=log_results,
                        help='Whether to log metrics.')
    parser.add_argument('-v', '--verbose', dest='verbose', type=bool, default=verbose,
                        help='Whether to use verbose text output.')
    
    args = parser.parse_args()
    
    #Other Plotting Args (too lazy to make command line args)
    args.show_plots = show_plots
    args.create_err_plot = create_err_plot
    args.create_scale_plot = create_scale_plot
    args.error_to_plot = error_to_plot
    args.err_plot_name = err_plot_name
    args.scale_plot_name = scale_plot_name
    args.figsize = figsize_inches
    args.units = error_units

    # Make sure output directory is selected and created.
    if args.output_dir is None:
        args.output_dir = util.select_directory()
        if args.output_dir is None:
            print("No output directory selected. Defaulting to current directory.")
            args.output_dir = "."
    
    if not os.path.exists(args.output_dir):
        input(f"Create output directory {args.output_dir}? (Ctrl+C to cancel)")
        os.makedirs(args.output_dir)
        print(f"Created Output Dir: {args.output_dir}")

    args.log_file = os.path.join(args.output_dir,args.log_name)

    # PROCESS BATCH =========================================================================
    # data_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens"
    # csv_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    # eval_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/teal-farsight-video.csv"
    # print(f"Maps:\n{[eval_file]}")
    # # for map in [csv_list[0]]:
    # for map in [os.path.basename(eval_file)]:
    #     args.pts2_file = join(data_dir,map)
    #     map_name = map.split(".")[0]
    #     args.map_image_file = join(data_dir,map_name+".png")
    #     args.err_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"{map_name}-point-errors"))
    #     args.scale_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"{map_name}-scale-factors"))
    #     name_split = map_name.split("-")
    #     args.test_name = f"{name_split[0].capitalize()} - {name_split[1].capitalize()} {name_split[2]}"
    #     args.test_note = f""



    args.use_headers = use_headers
    # LOAD POINTS FILE 1
    if args.pts1_file is None:
        args.pts1_file = csv_loader.open_csv_dialog('../data')
        if args.pts1_file is None:
            print("Ground Truth points file must be selected.")
            exit()
    print("\nLoading Point file 1...")
    headers1, data_gt = csv_loader.read_csv_points(args.pts1_file,args.use_headers,verbose=args.verbose)
    args.gt_file = os.path.basename(args.pts1_file)
    args.data_gt = csv_loader.fix_data_types(data_gt,set_str=['label'],set_float=['x','y','z'])
    # LOAD POINTS FILE 2
    if args.pts2_file is None:
        args.pts2_file = csv_loader.open_csv_dialog('../data')
        if args.pts2_file is None:
            print("Evaluation points file must be selected.")
            exit()
    print("\nLoading Point file 2...")
    read_robocup_format = True
    if not read_robocup_format:
        headers2, data_eval = csv_loader.read_csv_points(args.pts2_file,args.use_headers,verbose=args.verbose)
        args.eval_file = os.path.basename(args.pts2_file)
        args.data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])
    else:
        # MODIFIED FOR ROBOCUP GERMAN OPEN (Loads data correctly from specified Robocup csv format)
        headers2, data_eval = csv_loader.read_csv_points(args.pts2_file,headers = None,skip_rows=8,verbose=True)
        args.eval_file = os.path.basename(args.pts2_file)
        data_eval = csv_loader.append_concatenated_header(data_eval,'label',['type','name'],True)
        data_eval = csv_loader.extract_fields(data_eval,['label','x','y','z'])
        args.data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])
        detections_out = os.path.dirname(pts2_file)
        csv_loader.write_csv(join(detections_out,"FIXED-"+args.eval_file),['label','x','y','z'],data_eval)
    
    # LOAD MAP IMAGE
    args.map_image = None
    if args.map_image_file == "choose":
        args.map_image_file = util.open_image_dialog()
    if args.map_image_file:
        args.map_image = cv.imread(args.map_image_file)
        args.map_image = cv.cvtColor(args.map_image,cv.COLOR_BGR2RGB)

    args.scoring_threshold = config["scoring_threshold"]
    args.total_points = config["robocup_total_points"]
    # Pause before calculation
    input("Continue?: Ctrl+C to exit")
    
    main(args)
    
        