import open3d as o3d
import numpy as np
from os.path import join
import tkinter as tk
from tkinter import filedialog
import utilities as util
import calculate_metrics as cm

def select_point_cloud_files(root_dir, select="Point Cloud File"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    files = filedialog.askopenfilename(
        initialdir=root_dir,
        title=f"Select {select}",
        filetypes=[("Point Cloud Files", "*.ply *.pcd *.xyz *.las *.laz")]
    )
    print (f"{select}: {files}")
    return files

def analyze_pointcloud(pcd,cloud_name=None):
    """
    Analyze a point cloud and print useful metrics.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to analyze.
    
    Returns:
        dict: A dictionary of computed metrics.
    """
    if cloud_name is not None:
        print(f"Getting Cloud Information for: {cloud_name}...")

    metrics = {}

    # Number of points
    metrics["num_points"] = np.asarray(pcd.points).shape[0]
    
    # Fields present
    metrics["has_normals"] = pcd.has_normals()
    metrics["has_colors"] = pcd.has_colors()

    # Bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    metrics["bounding_box_extent"] = aabb.get_extent()
    metrics["bounding_box_volume"] = np.prod(aabb.get_extent())
    metrics['bounding_box_center'] = aabb.get_center()
    
    # Center of mass (mean of all points)
    points = np.asarray(pcd.points)
    metrics["center_of_mass"] = np.mean(points, axis=0)
    
    # Density (approximation: points per volume of bounding box)
    metrics["density"] = metrics["num_points"] / metrics["bounding_box_volume"]

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
        metrics["convex_hull_density"] = metrics["num_points"] / hull_volume if hull_volume > 0 else None
    except Exception as e:
        metrics["convex_hull_volume"] = None
        metrics["convex_hull_density"] = None
        print(f"Warning: Could not compute convex hull. Reason: {e}")

    # Print all metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("")
    return metrics

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud from {file_path}")
    return pcd

def compute_cloud_to_cloud_distance(source_pcd, target_pcd):
    distances = source_pcd.compute_point_cloud_distance(target_pcd)
    return np.array(distances)

# def compute_completeness(ground_truth_pcd, eval_pcd, threshold):
def compute_completeness(distances, threshold):
    # distances = compute_cloud_to_cloud_distance(ground_truth_pcd, eval_pcd)
    within_threshold = np.sum(distances < threshold)
    completeness = (within_threshold / len(distances)) * 100.0
    return completeness


if __name__ == "__main__":
    # VARIABLES
    root_dir = "C:\\Users\\nullp\\Documents\\RoboCup24"
    ground_truth_path = join(root_dir,"labyrinth.pcd")
    # eval_map_path = join(root_dir,"test1.ply")
    eval_map_path = select_point_cloud_files(root_dir)

    distance_threshold = 0.05  # Adjust this threshold based on your use case

    # Log Variables 
    log_results = True
    output_dir = root_dir
    log_file = join(output_dir,"cloud_metric_log.csv")

    verbose = True

    # LOAD IN CLOUDS
    # Load Ground Truth Cloud, print info
    ground_truth_pcd = load_point_cloud(ground_truth_path)
    print(f"Ground Truth Loaded: {ground_truth_path}")
    gt_info = analyze_pointcloud(ground_truth_pcd,"Ground Truth")
    # Load Evaluation Cloud, print info
    eval_map_pcd = load_point_cloud(eval_map_path)
    print(f"Evaluation Cloud Loaded: {eval_map_path}")
    eval_info = analyze_pointcloud(eval_map_pcd,"Evaluation Map")

    # COMPUTE METRICS
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
    # completeness = compute_completeness(ground_truth_pcd, eval_map_pcd, distance_threshold)
    completeness = compute_completeness(distances2,distance_threshold)
    print(f"Completeness: {completeness:.2f}% of ground truth points within {distance_threshold} units.")

    if log_results:
        headers = ["Timestamp","GT File","Eval File",
                   "Has Color","Has Normals","Total Points",
                   "BB Extent","BB Volume","BB Density","CH Volume","CH Density",
                    "Dist GT->EV","Dist EV->GT","Dist Avg","Completeness","CompDist Threshold"]
        log_data = [util.timestamp(),ground_truth_path,eval_map_path,
                    eval_info["has_colors"],eval_info["has_normals"],eval_info["num_points"],
                    eval_info["bounding_box_extent"],eval_info["bounding_box_volume"],eval_info["density"],eval_info["convex_hull_volume"],eval_info["convex_hull_density"],
                    d1_mean,d2_mean,d_avg,completeness,distance_threshold]
        cm.log_to_csv(log_file,log_data,headers,verbose)