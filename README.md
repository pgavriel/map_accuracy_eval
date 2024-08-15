# map_accuracy_eval   

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
map_eval_2d.py - Script for performing 2d evaluations  
map_eval_3d.py - Script for performing 3d evaluations  
  

**ARCHIVED SCRIPTS**  
[ARCHIVED] alignment_test.py - Testing an automatic full alignment of points. Should be contained in point_manip  
  
[ARCHIVED] assign_markers3.py - LEGACY: Looks like the first all-in-one script for assigning points and generating metrics  
  
[ARCHIVED] gui_test.py - Testing out using TKinter to make a more sophisticated GUI  
  
[ARCHIVED] gui_test2.py - Similar to gui_test.py, testing keybinds  
  
[ARCHIVED] map_evaluation.py - LEGACY: Seems like what came after assign_markers3.py  
  
[ARCHIVED] point_matching.py - LEGACY: Provides functions for translating, rotating, and scaling sets of points, as well as functions for doing distance calculations between sets of points and generating metrics    
