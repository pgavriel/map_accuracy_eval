import point_manip as pm
import csv_loader
import utilities as util
import calculate_metrics as metrics

if __name__ == "__main__":
    # TODO: Implement argparse, test name, output_dir, gt_file, eval_file

    # LOAD IN YOUR POINTS
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
    # CALCULATE COVERAGE
    data_gt, data_eval, coverage = metrics.calc_coverage(data_gt,data_eval)

