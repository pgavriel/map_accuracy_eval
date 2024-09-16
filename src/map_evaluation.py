import os
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
    metric['cvg_found'] = int(metric['cvg_total'] * metric['coverage'] / 100)
    # CALCULATE GLOBAL ERROR METRICS
    metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    # CALCULATE SCALE FACTOR METRICS
    metric['point_scales'],metric['scale_avg'],metric['scale_std'] = metrics.calc_scale_factors(data_gt, data_eval,use_z=args.eval_3d,verbose=args.verbose)
    metric['norm_scale_std'] = metric['scale_std'] / metric['scale_avg']

    if auto_scale:
        # AUTO SCALE AND FIND ERROR METRICS AGAIN
        print(f"Scaling Eval Points based on found scale factor average ({metric['scale_avg']})...")
        scaled_data_eval = pm.scale_points_wrt_origin(data_eval,metric['scale_avg'],args.eval_3d)
        # CALCULATE GLOBAL ERROR METRICS
        metric['scaled_point_errors'],metric['scaled_error_avg'],metric['scaled_error_std'] = metrics.calc_error(data_gt, scaled_data_eval,use_z=args.eval_3d,verbose=args.verbose)
    
    # GENERATE OUTPUT PLOTS
    if args.create_plots and not args.eval_3d:
        if args.create_err_plot:
            metrics.generate_pointerror_contour_plot(data_eval,metric[args.error_to_plot],metric,args.map_image,args.err_plot_name,args.show_plots,args.figsize)
        if args.create_scale_plot:
            metrics.generate_scalefactor_plot(data_eval,metric,args.exclude_std_devs,args.map_image,args.scale_plot_name,args.show_plots,args.figsize)
    
    # LOG RESULTS
    if log_results:
        # Write metrics results to specified CSV file.
        if eval_3d:
            args.test_note = args.test_note + " (3D Evaluation)"
        log_list = [util.timestamp(),args.test_name,args.test_note,
                    metric['gt_file'],metric['eval_file'],
                    metric['coverage'],
                    metric['error_avg'],metric['error_std'],
                    metric['scale_avg'],metric['scale_std'],metric['norm_scale_std'],
                    metric['scaled_error_avg'],metric['scaled_error_std']]
        metrics.log_to_csv(args.log_file, log_list,verbose=args.verbose)


if __name__ == "__main__":
    # DEFAULT ARGUMENTS 
    verbose = False
    log_results = True
    eval_3d = False
    auto_scale = True

    # output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/point_method"
    default_output = "C:/Users/nullp/Projects/map_accuracy_eval/output/ua5/"
    # output_dir = util.select_directory(default_output, "Select output dir")
    output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/test"
    log_name = "output_log.csv"

    test_name = "Test"
    test_note = "3rd Floor"

    gt_2f_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/GroundTruth/map_pts_2f.csv"
    gt_3f_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/GroundTruth/map_pts_3f.csv"

    # pts1_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test3.csv"
    pts1_file = gt_2f_file

    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/"
    # pts2_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test5.csv"
    # pts2_file = pts1_file
    pts2_file = csv_loader.open_csv_dialog(d)

    use_headers = None    # Set to None if first line of csv has headers
    # use_headers = ['label','x','y']
    # if eval_3d: use_headers.append('z')
    
    d = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/"
    map_image_file = util.open_image_dialog(d)
    
    create_plots = True
    figsize_inches = (16,8) # (8,8) default
    show_plots = True
    create_err_plot = True
    create_scale_plot = True
    error_to_plot = 'point_errors' # or 'scaled_point_errors'
    # error_to_plot = 'scaled_point_errors' # or 'scaled_point_errors'
    exclude_std_devs = 2
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

    
    main(args)
    
        