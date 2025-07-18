# map_accuracy_eval   
## **UNDER ACTIVE DEVELOPMENT**  
This package provides a toolkit for evaluating the accuracy between a map created by a robotic system, and a ground truth map. This is primarily done by identifying corresponding points in both the ground truth and evaluation map, giving each correspondance a unique label, and storing this information in separate CSV files with the fields [label, x, y] or [label, x, y, z] for 3D evaluations. Those CSV files can then be loaded in by the toolkit and metrics can be calculated.    
  
**RUNNABLE SCRIPTS**  
* **cloud_to_cloud_evaluation.pyy** - (3D) Directly compares two 3D pointclouds which need to already be aligned with each other. Prints out some basic information for each cloud for comparison, calculates the distance from ground truth to eval as well as evel to ground truth, and calculates Completeness/Coverage of the ground truth map within a specified tolerance. Optionally saves a new point cloud containing all outlier points to visualized the areas of the ground truth which are not covered. Reads in the run configuration from cloud_eval_config.json so the script does not need to be changed. Setting either input file to 'null' will prompt you with a file dialogue to select each input cloud.    
* **reference_point_evaluation.py** -  (2D/3D) This script takes in two sets of labelled points (and optionally a map background image), calculates and logs all relevant metrics between the two points, and generates plots to visualize the metrics when doing 2D evaluations. It functions exactly the same for 3D evaluations, but doesn't generate any plots, only metrics. These metrics are global error and scale factors; they score the maps accuracy holistically.  
* **map_labeller.py** - (2D) Script is used to label 2D evaluation points within a map image. Points can easily be imported, edited, and exported in a format that can be used by the evaluation tool. *NOTE: For 3D evaluations, a tool like CloudCompare can be used to export labelled 3D point lists.*   
* **fiducial_evaluation.py** - (2D) This tool is used for performing fiducial evaluations (currently either cylinder or splitcross type). Fiducials are artifacts placed within the mapping area to assess local errors within the map.    
* **align_pts_to_map.py** - (2D) Tool for aligning reference points to a corresponding map image and saving the new transformed points (for visualizing)  
  
**METRICS**  
* Coverage - The percentage of labelled ground truth points contained in the list of evaluation points.    
* Error Average (and Std Dev) - The average of all point errors. Point error is the average absolute difference (between ground truth and evaluation points) in distance between a given point and every other point being evaluated.    
* Scale Factor Average (and Std Dev) - Similar to the Error Average calculation, but instead of comparing absolute differences in distance, the relative difference in distance are expressed as a scaling ratio (GT Distance / Eval Distance).  
* Scaled Error Average (and Std Dev) - The evaluation points are scaled by the factor determined by the Scale Factor Average metric, and the Error metric is recomputed.  
* Fiducial Score - An alternate evaluation method which utilizes artificual fiducials placed in the mapping environment. Fiducials are evaluated on a pass/fail basis, based on specified criteria for distance and angle tolerance with respect to the ground truth. Expressed as a percentage of passing fiducials.  
     
**HELPER FILES**  
* calculate_metrics.py - Contains all functions for calculating metrics AND generating plots  
  
* csv_loader.py - Provides functions for loading and reformatting csv files, which can then be used as reference points for 2D and 3D evaluations    
  
* point_manip.py - Contains all functions for manipulating points objects. Translate, rotate, scale, align, etc.  
   
* utilities.py - Provides various functions for writing files, drawing on images, etc.  
  
