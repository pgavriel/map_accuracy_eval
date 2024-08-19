import point_manip as pm
import csv_loader
import utilities as util
import calculate_metrics as metrics

if __name__ == "__main__":
    # TODO: Implement argparse, test name, output_dir, gt_file, eval_file

    # LOAD IN YOUR POINTS
    log_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/output_log.csv"
    pts1_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test1.csv"
    pts2_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/metric_test2.csv"

    headers1, data_gt = csv_loader.read_csv_points(pts1_file)
    data_gt = csv_loader.fix_data_types(data_gt,set_str=['label'],set_float=['x','y'])
    headers2, data_eval = csv_loader.read_csv_points(pts2_file)
    data_eval = csv_loader.fix_data_types(data_eval,set_str=['label'],set_float=['x','y'])

    # LOAD MAP IMAGES
    # TODO: Add option to include map image(s) that line up with the points, set the blend, used to make figures later

    # Do points need to be scaled? 

    # CALCULATE METRICS ============================================
    verbose = True
    write_results = True
    eval_3d = False
    test_name = "Experiment 2"
    test_note = "New distance function"
    metric = metrics.initialize_metrics()
    # CALCULATE COVERAGE
    data_gt, data_eval, metric['coverage'] = metrics.calc_coverage(data_gt,data_eval,verbose=verbose)
    # CALCULATE GLOBAL ERROR METRICS
    metric['point_errors'],metric['error_avg'],metric['error_std'] = metrics.calc_error(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    # CALCULATE SCALE FACTOR METRICS
    metric['point_scales'],metric['scale_avg'],metric['scale_std'] = metrics.calc_scale_factors(data_gt, data_eval,use_z=eval_3d,verbose=verbose)
    metric['norm_scale_std'] = metric['scale_std'] / metric['scale_avg']

    # LOG RESULTS
    if write_results:
        # Write metrics results to specified CSV file.
        if eval_3d:
            test_note.append(" (3D Evaluation)")
        log_list = [util.timestamp(),test_name,test_note,
                    metric['coverage'],
                    metric['error_avg'],metric['error_std'],
                    metric['scale_avg'],metric['scale_std'],metric['norm_scale_std']]
        metrics.log_to_csv(log_file, log_list,verbose)
        