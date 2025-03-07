import os
from os.path import join
import cv2 as cv
import argparse
import point_manip as pm
import csv_loader
import utilities as util
import calculate_metrics as metrics

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
        scaled_map_image = util.scale_image_with_aspect_ratio(args.map_image,metric['scale_avg'])
        metric['scaled_point_scales'],metric['scaled_scale_avg'],metric['scaled_scale_std'] = metrics.calc_scale_factors(data_gt, scaled_data_eval,use_z=args.eval_3d,verbose=args.verbose)
        metric['scaled_norm_scale_std'] = metric['scaled_scale_std'] / metric['scaled_scale_avg']

    # UNIT SCALING
    metric_scaler = 0.033311475409836 #1
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
                    metric['scaled_error_avg']*s,metric['scaled_error_std']*s]
        metrics.log_to_csv(args.log_file, log_list,verbose=args.verbose)


if __name__ == "__main__":
    # DEFAULT ARGUMENTS 
    verbose = False
    log_results = True
    eval_3d = False
    auto_scale = False

    # output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/point_method"
    default_output = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/"
    output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/output"
    # output_dir = util.select_directory(default_output, "Select output dir")
    log_name = "2d_output_log.csv"

    test_name = "NewHorizon 3F (RESCALED)"
    test_note = "3F All Points (2D Eval)"
    # Structural Reference Points  -  C Targets  -  All Points

    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens" # Default dir
    gt_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/gt/GPS-ground-truth.csv"
    # pts1_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test3.csv"
    # pts1_file = csv_loader.open_csv_dialog(d)
    pts1_file = gt_file

    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens"
    pts2_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/parrot-farsight-photo.csv"
    # pts2_file = pts1_file
    # pts2_file = csv_loader.open_csv_dialog(d)

    use_headers = None    # Set to None if first line of csv has headers
    # use_headers = ['label','x','y','z']
    # if eval_3d: use_headers.append('z')   
    
    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens"
    map_image_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/parrot-farsight-photo.png"
    # map_image_file = util.open_image_dialog(d)
    # map_image_file = None
    
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
    data_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens"
    csv_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    eval_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/devens/teal-farsight-video.csv"
    print(f"Maps:\n{[eval_file]}")
    # for map in [csv_list[0]]:
    for map in [os.path.basename(eval_file)]:
        args.pts2_file = join(data_dir,map)
        map_name = map.split(".")[0]
        args.map_image_file = join(data_dir,map_name+".png")
        args.err_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"{map_name}-point-errors"))
        args.scale_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"{map_name}-scale-factors"))
        name_split = map_name.split("-")
        args.test_name = f"{name_split[0].capitalize()} - {name_split[1].capitalize()} {name_split[2]}"
        args.test_note = f""



        args.use_headers = use_headers
        # LOAD POINTS 
        if args.pts1_file is None:
            args.pts1_file = csv_loader.open_csv_dialog('../data')
            if args.pts1_file is None:
                print("Ground Truth points file must be selected.")
                exit()
        headers1, data_gt = csv_loader.read_csv_points(args.pts1_file,args.use_headers,verbose=args.verbose)
        args.gt_file = os.path.basename(args.pts1_file)
        args.data_gt = csv_loader.fix_data_types(data_gt,set_str=['label'],set_float=['x','y','z'])

        if args.pts2_file is None:
            args.pts2_file = csv_loader.open_csv_dialog('../data')
            if args.pts2_file is None:
                print("Evaluation points file must be selected.")
                exit()
        headers2, data_eval = csv_loader.read_csv_points(args.pts2_file,args.use_headers,verbose=args.verbose)
        args.eval_file = os.path.basename(args.pts2_file)
        args.data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])
        
        # LOAD MAP IMAGE
        args.map_image = None
        if args.map_image_file == "choose":
            args.map_image_file = util.open_image_dialog()
        if args.map_image_file:
            args.map_image = cv.imread(args.map_image_file)
            args.map_image = cv.cvtColor(args.map_image,cv.COLOR_BGR2RGB)

        
        
        main(args)
    
        