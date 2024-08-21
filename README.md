# map_accuracy_eval   
## **UNDER ACTIVE DEVELOPMENT**  
#### **DOCUMENTATION IS UNDER CONSTRUCTION**  
This package provides a toolkit for evaluating the accuracy between a map created by a robotic system, and a ground truth map. This is primarily done by identifying corresponding points in both the ground truth and evaluation map, giving each correspondance a unique label, and storing this information in separate CSV files with the fields [label, x, y] or [label, x, y, z] for 3D evaluations. Those CSV files can then be loaded in by the toolkit and metrics can be calculated.    
**USEAGE**  
*   
  
**METRICS**  
* Coverage - The percentage of labelled ground truth points contained in the list of evaluation points.    
* Error Average (and Std Dev) - The average of all point errors. Point error is the average absolute difference (between ground truth and evaluation points) in distance between a given point and every other point being evaluated.    
* Scale Factor Average (and Std Dev) - Similar to the Error Average calculation, but instead of comparing absolute differences in distance, the relative difference in distance are expressed as a scaling ratio (GT Distance / Eval Distance).  
* Scaled Error Average (and Std Dev) - The evaluation points are scaled by the factor determined by the Scale Factor Average metric, and the Error metric is recomputed.  
* Fiducial Score - An alternate evaluation method which utilizes artificual fiducials placed in the mapping environment. Fiducials are evaluated on a pass/fail basis, based on specified criteria for distance and angle tolerance with respect to the ground truth. Expressed as a percentage of passing fiducials.  
     
**PYTHON SCRIPTS**  
* align_pts_to_map.py - Tool for aligning reference points to a corresponding map image and saving the new transformed points (for visualizing)  
  
* calculate_metrics.py - Contains all functions for calculating metrics AND generating plots?  
  
* csv_loader.py - Provides functions for loading and reformatting csv files, which can then be used as reference points for 2D and 3D evaluations    

* map_labeller.py - Tool for labelling 2D points on a map image.  
  
* point_manip.py - Contains all functions for manipulating points objects. Translate, rotate, scale, align, etc.  
   
* utilities.py - Provides various functions for writing files, drawing on images, and visualizing metrics?  
  
* viewport_tool.py - UNDER CONSTRUCTION: Supposed to provide a more precise way to label images, especially large ones. Should eventually be integrated into or used by the map labeller.  
  
TODO:   
align_pts_to_pts.py - Tool for aligning a set of reference points to ground truth points (for visualizing)  
align_map_to_map.py - Tool for aligning two map images (for visualizing)  
map_eval_2d.py - Script for performing 2d reference point evaluations  
map_eval_3d.py - Script for performing 3d reference point evaluations  
map_eval_fiducial.py -  Script for evaluating fidicual pairs (cylinder or x shaped) in a pass/fail manner.  
Tool for assessing the density of reference points to help spread them evenly in a map. Should be a feature of map_labeller?

**ARCHIVED SCRIPTS**  
[ARCHIVED] alignment_test.py - Testing an automatic full alignment of points. Should be contained in point_manip  
  
[ARCHIVED] assign_markers3.py - LEGACY: Looks like the first all-in-one script for assigning points and generating metrics  
  
[ARCHIVED] gui_test.py - Testing out using TKinter to make a more sophisticated GUI  
  
[ARCHIVED] gui_test2.py - Similar to gui_test.py, testing keybinds  
  
[ARCHIVED] map_evaluation.py - LEGACY: Seems like what came after assign_markers3.py  
  
[ARCHIVED] point_matching.py - LEGACY: Provides functions for translating, rotating, and scaling sets of points, as well as functions for doing distance calculations between sets of points and generating metrics    
