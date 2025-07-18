from os.path import join, dirname, basename, abspath, exists
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import utilities as util
import calculate_metrics as cm

def load_config(config_path=None,verbose=True):
    """
    Loads and validates configuration from a JSON file one directory above the script.
    """
    if config_path is None:
        script_dir = dirname(abspath(__file__))
        config_path = join(script_dir, "..", "cloud_eval_config.json")

    print(f"Loading Config: {config_path}")

    if not exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Required keys and their expected types
    required_keys = {
        "root_directory": str,
        "ground_truth_file": (str, type(None)),  # Allows None (null)
        "eval_file": (str, type(None)),  # Allows None (null)
        "event_name": str,
        "note": str,
        "dist_threshold": float,
        "log_file": str,
        "log_results": bool,
        "save_outliers": bool,
        "verbose": bool
    }
    
    # Check for missing keys and type validation
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Config key '{key}' must be of type {expected_type}, but got {type(config[key])}")

    if verbose:
        print("Configuration Loaded: ")
        print(json.dumps(config, indent=4))

    return config

def select_point_cloud_files(root_dir, select="Point Cloud File"):
    """
    Uses Tkinter to open a file selection dialogue for a point cloud file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    files = filedialog.askopenfilename(
        initialdir=root_dir,
        title=f"Select {select}",
        filetypes=[("Point Cloud Files", "*.ply *.pcd *.xyz *.las *.laz")]
    )
    print (f"{select}: {files}")
    return files

def analyze_pointcloud(pcd,cloud_name=None, verbose=False):
    """
    Analyze a point cloud and optionally print useful metrics.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to analyze.
        cloud_name (str): Name identifier for cloud being analyzed.
        verbose (bool): Whether to automatically print all metrics.
    Returns:
        dict: A dictionary of computed metrics.
    """
    if cloud_name is not None:
        print(f"Getting Cloud Information for: {cloud_name}...")

    metrics = {}
    # Fields present
    metrics["Has Normals"] = pcd.has_normals()
    metrics["Has Colors"] = pcd.has_colors()

    # Number of points
    metrics["Total Points"] = np.asarray(pcd.points).shape[0]
    
    aabb = pcd.get_axis_aligned_bounding_box()
    metrics["Bounding Box Volume"] = np.prod(aabb.get_extent())

    # Density (approximation: points per volume of bounding box)
    metrics["Density"] = metrics["Total Points"] / metrics["Bounding Box Volume"]
    # Bounding box
    
    bounding_box_extent = aabb.get_extent()
    metrics["Bound Extent X"] = bounding_box_extent[0]
    metrics["Bound Extent Y"] = bounding_box_extent[1]
    metrics["Bound Extent Z"] = bounding_box_extent[2]
    # metrics['bounding_box_center'] = aabb.get_center()
    
    # Center of mass (mean of all points)
    points = np.asarray(pcd.points)
    center_of_mass = np.mean(points, axis=0)
    metrics["Center Mass X"] = center_of_mass[0]
    metrics["Center Mass Y"] = center_of_mass[1]
    metrics["Center Mass Z"] = center_of_mass[2]

    # Principal axes and orientation
    # covariance = np.cov(points, rowvar=False)
    # eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # metrics["principal_axes"] = eigenvectors
    # metrics["eigenvalues"] = eigenvalues

    # Radius and volume of convex hull (optional, computationally intensive)
    try:
        hull = pcd.compute_convex_hull()[0]
        hull_volume = hull.get_volume()
        metrics["convex_hull_volume"] = hull_volume
        metrics["convex_hull_density"] = metrics["Total Points"] / hull_volume if hull_volume > 0 else None
    except Exception as e:
        metrics["convex_hull_volume"] = None
        metrics["convex_hull_density"] = None
        print(f"Warning: Could not compute convex hull. Reason: {e}")

    if verbose:
        # Print all metrics
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("")
    return metrics

def compare_cloud_metrics(pcd_gt, pcd_eval,gt_name="Ground Truth Cloud",eval_name="Eval Cloud"):
    metrics1 = analyze_pointcloud(pcd_gt, gt_name)
    metrics2 = analyze_pointcloud(pcd_eval, eval_name)
    headers = ["Metric",gt_name, eval_name,"Diff"]
    print(f"\n[{headers[0].center(30)}][{headers[1].center(30)}][{headers[2].center(30)}][{headers[3].center(15)}]")
    print("=================================================================================================================")
    for m1, m2 in zip(metrics1,metrics2):
        assert m1 == m2
        v1 = metrics1[m1]
        v2 = metrics2[m2]
        diff_str = None
        s = ["","+"]
        if m1 != "Has Normals" and m1 != "Has Colors":
            diff = ((v2/v1)-1) * 100
            diff_str = f"{s[0] if diff < 0 else s[1]}{diff:.1f}%"
        else:
            if v1 == v2:
                diff_str = "=="
            else:
                diff_str = "!="
        print_str = f"[{str(m1).center(30)}][{str(v1).center(30)}][{str(v2).center(30)}]"
        if diff_str is not None:
            print_str += f"[{diff_str.center(15)}]"
        print(print_str)
    
    return metrics1, metrics2

def load_point_cloud(file_path):
    """
    Read pointcloud using Open3D
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud from {file_path}")
    return pcd

def compute_cloud_to_cloud_distance(source_pcd, target_pcd):
    """
    For each point in source_pcd, find the minimum distance to a point
    in target_pcd, and assemble these minimum distances into an array.
    [Get Distances from SOURCE to TARGET]
    """
    distances = source_pcd.compute_point_cloud_distance(target_pcd)
    return np.array(distances)

# def compute_completeness(ground_truth_pcd, eval_pcd, threshold):
def compute_completeness(ground_truth_pcd, distances, threshold):
    """
    Computes the completeness of coverage for a evaluation point cloud
    to a ground truth cloud within a specified threshold.
    The distances (GT -> Eval) should be computed beforehand.

    Args:
        ground_truth_pcd (o3d.geometry.PointCloud): Ground truth cloud to check against
        distances (array): Array of min distances from each GT point to the eval cloud
        threshold (float): All minimum distances >= threshold = outlier
    """
    # Identify points that fall outside the threshold
    outside_indices = np.where(distances >= threshold)[0]
    
    # Create a new point cloud with only those points
    incomplete_pcd = o3d.geometry.PointCloud()
    incomplete_pcd.points = o3d.utility.Vector3dVector(np.asarray(ground_truth_pcd.points)[outside_indices])

    # Compute completeness
    completeness = ((len(distances) - len(outside_indices)) / len(distances)) * 100.0

    return completeness, incomplete_pcd


if __name__ == "__main__":
    # LOAD CONFIGURATION ===================================================
    config = load_config()
    # ASSIGN VARIABLES
    root_dir = config["root_directory"]
    # Establish Ground Truth File
    if config["ground_truth_file"] is None:
        ground_truth_path = select_point_cloud_files(root_dir, "Ground Truth Point Cloud")
    else:
        ground_truth_path = join(root_dir, config["ground_truth_file"])
    # Establish Evaluation File and outliers output file
    if config["eval_file"] is None:
        eval_map_path = select_point_cloud_files(root_dir, "Evaluation Point Cloud")
    else:
        eval_map_path = join(root_dir, config["eval_file"])
    eval_map_file = basename(eval_map_path).split(".")[0]
    output_incomplete_path = join(dirname(eval_map_path),eval_map_file+"_missing_points.ply")

    event_name = config["event_name"]
    note = config["note"]
    distance_threshold = config["dist_threshold"] # Adjust this threshold based on your use case

    log_results = config["log_results"]
    log_file = join(root_dir, config["log_file"])
    save_outliers = config["save_outliers"]
    verbose = config["verbose"]


    # LOAD IN CLOUDS ===========================================================
    # Load Ground Truth Cloud, print info
    ground_truth_pcd = load_point_cloud(ground_truth_path)
    print(f"Ground Truth Loaded: {ground_truth_path}")
    # Load Evaluation Cloud, print info
    eval_map_pcd = load_point_cloud(eval_map_path)
    print(f"Evaluation Cloud Loaded: {eval_map_path}")
    gt_info, eval_info = compare_cloud_metrics(ground_truth_pcd,eval_map_pcd)

    # COMPUTE METRICS ===========================================================
    # Quality: Cloud-to-cloud distance (Eval -> GT)
    print(f"Computing Eval -> GT Distances...")
    distances1 = compute_cloud_to_cloud_distance(eval_map_pcd, ground_truth_pcd)
    print(f"Cloud-to-cloud distances (Eval -> GT):")
    d1_mean = np.mean(distances1)
    print(f"Mean distance: {d1_mean:.4f}")
    print(f"Max distance: {np.max(distances1):.4f}")
    print(f"Min distance: {np.min(distances1):.4f}\n")

    # Quality: Cloud-to-cloud distance (GT -> Eval)
    print(f"Computing GT -> Eval Distances...")
    distances2 = compute_cloud_to_cloud_distance(ground_truth_pcd, eval_map_pcd)
    print(f"Cloud-to-cloud distances (GT -> Eval):")
    d2_mean = np.mean(distances2)
    print(f"Mean distance: {d2_mean:.4f}")
    print(f"Max distance: {np.max(distances2):.4f}")
    print(f"Min distance: {np.min(distances2):.4f}\n")

    d_avg = np.mean([d1_mean,d2_mean])
    print(f"Distance Average (Chamfer distance): {d_avg:.4f}\n")

    # Completeness: Percentage of ground truth points within distance threshold
    completeness, incomplete_pcd = compute_completeness(ground_truth_pcd,distances2,distance_threshold)
    print(f"Completeness: {completeness:.2f}% of ground truth points within {distance_threshold} units.")
    
    # Compute Mapping Score
    map_score = (completeness) / (1 + d1_mean)
    print(f"\n3D MAP SCORE (C / (1+E)): {map_score:.2f}\n")

    # SAVE OUTPUTS ===========================================================
    if save_outliers:
        # Save and visualize missing points
        o3d.io.write_point_cloud(output_incomplete_path, incomplete_pcd)
        print(f"Saved incomplete/missing points to {output_incomplete_path}")

    if log_results:
        headers = ["Timestamp","Event","Note","GT File","Eval File",
                   "Has Color","Has Normals","Total Points",
                   "BB Extent","BB Volume","BB Density","CH Volume","CH Density",
                    "Dist GT->EV","Dist EV->GT","Dist Avg (Chamfer Distance)","Completeness","CompDist Threshold","Map Score"]
        
        bb_extent = [eval_info["Bound Extent X"],eval_info["Bound Extent Y"],eval_info["Bound Extent Z"]]
        log_data = [util.timestamp(),event_name,note,basename(ground_truth_path),basename(eval_map_path),
                    eval_info["Has Colors"],eval_info["Has Normals"],eval_info["Total Points"],
                    bb_extent,eval_info["Bounding Box Volume"],eval_info["Density"],eval_info["convex_hull_volume"],eval_info["convex_hull_density"],
                    d1_mean,d2_mean,d_avg,completeness,distance_threshold,map_score]
        cm.log_to_csv(log_file,log_data,headers,verbose)