import os
import cv2 as cv
import point_manip as pm
import csv_loader
import utilities as util
import calculate_metrics as metrics

if __name__ == "__main__":
    # TODO: Implement argparse, test name, output_dir, gt_file, eval_file
    verbose = True
    write_results = True
    eval_3d = False
    auto_scale = True

    output_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/dev-testing"
    log_name = "output_log.csv"
    log_file = os.path.join(output_dir,log_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created Output Dir: {output_dir}")

    create_plots = True
    show_plots = True
    create_err_plot = True
    create_scale_plot = True
    error_to_use = 'point_errors' # or 'scaled_point_errors'
    err_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"point-errors"))
    # err_plot_name = None
    exclude_std_devs = 2
    scale_plot_name = os.path.join(output_dir,util.generate_unique_filename(f"scale-factors"))
    # scale_plot_name = None

    test_name = "Plot Experiments"
    test_note = "Dev"

    # LOAD IN YOUR POINTS
    
    pts1_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test3.csv"
    # pts1_file = csv_loader.open_csv_dialog('../data')
    pts2_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test5.csv"
    # pts2_file = pts1_file
    

    use_headers = None    # Set to None if first line of csv has headers
    # use_headers = ['label','x','y']
    # if eval_3d: use_headers.append('z')
    headers1, data_gt = csv_loader.read_csv_points(pts1_file,use_headers,verbose=verbose)
    gt_file = os.path.basename(pts1_file)
    data_gt = csv_loader.fix_data_types(data_gt,set_str=['label'],set_float=['x','y','z'])
    headers2, data_eval = csv_loader.read_csv_points(pts2_file,use_headers,verbose=verbose)
    eval_file = os.path.basename(pts2_file)
    data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y','z'])

    # LOAD MAP IMAGES
    # TODO: Add option to include map image(s) that line up with the points, set the blend, used to make figures later
    map_image = None
    # map_image_file = util.open_image_dialog()
    map_image_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_gt.png"
    if map_image_file:
        map_image = cv.imread(map_image_file)


    # Do points need to be scaled? 

    # CALCULATE METRICS ============================================
    metric = metrics.initialize_metrics()
    metric['test_name'] = test_name
    metric['test_note'] = test_note
    metric['gt_file'] = gt_file
    metric['eval_file'] = eval_file
    # CALCULATE COVERAGE
    data_gt, data_eval, metric['coverage'] = metrics.calc_coverage(data_gt,data_eval,verbose=verbose)
    # CALCULATE GLOBAL ERROR METRICS
    metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    # CALCULATE SCALE FACTOR METRICS
    metric['point_scales'],metric['scale_avg'],metric['scale_std'] = metrics.calc_scale_factors(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    metric['norm_scale_std'] = metric['scale_std'] / metric['scale_avg']

    if auto_scale:
        # AUTO SCALE AND FIND ERROR METRICS AGAIN
        print(f"Scaling Eval Points based on found scale factor average ({metric['scale_avg']})...")
        data_eval = pm.scale_points_wrt_origin(data_eval,metric['scale_avg'],eval_3d)
        # CALCULATE GLOBAL ERROR METRICS
        metric['scaled_point_errors'],metric['scaled_error_avg'],metric['scaled_error_std'] = metrics.calc_error(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    
    # GENERATE OUTPUT PLOTS
    if create_plots and not eval_3d:
        if create_err_plot:
            metrics.generate_pointerror_contour_plot(data_eval,metric[error_to_use],metric,map_image,err_plot_name,show_plots)
        if create_scale_plot:
            metrics.generate_scalefactor_plot(data_eval,metric,exclude_std_devs,map_image,scale_plot_name,show_plots)
    
    # LOG RESULTS
    if write_results:
        # Write metrics results to specified CSV file.
        if eval_3d:
            test_note = test_note + " (3D Evaluation)"
        log_list = [util.timestamp(),test_name,test_note,
                    metric['gt_file'],metric['eval_file'],
                    metric['coverage'],
                    metric['error_avg'],metric['error_std'],
                    metric['scale_avg'],metric['scale_std'],metric['norm_scale_std'],
                    metric['scaled_error_avg'],metric['scaled_error_std']]
        metrics.log_to_csv(log_file, log_list,verbose)
        