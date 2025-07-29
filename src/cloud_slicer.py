import os
from os.path import join, dirname, basename, abspath, exists
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cloud_to_cloud_evaluation import select_point_cloud_files, load_point_cloud
from glob import glob

def slice_pointcloud(pcd, position, thickness, axis='z'):
    """
    Slice a point cloud perpendicular to the given axis.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        position (float): Center position of the slice along the axis.
        thickness (float): Total thickness of the slice.
        axis (str): Axis to slice along ('x', 'y', or 'z').

    Returns:
        o3d.geometry.PointCloud: Sliced point cloud.
    """
    print(f"Slicing Cloud... [ Axis: {axis} | Center: {position:.2f} | Thickness: {thickness:.2f}]")
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    i = axis_map[axis.lower()]
    points = np.asarray(pcd.points)
    mask = np.abs(points[:, i] - position) <= thickness / 2.0
    sliced_pcd = pcd.select_by_index(np.where(mask)[0])
    return sliced_pcd

def rasterize_pointcloud(pcd, axis='z', width=512, height=512,
                         background_color=(1.0, 1.0, 1.0),
                         output_path="output.png",
                         text: str = None,
                         padding: int = 10,
                         xy_range=None):
    """
    Project and rasterize a point cloud along the given axis to a 2D image using OpenCV.

    Args:
        pcd (o3d.geometry.PointCloud): Sliced point cloud.
        axis (str): Axis to project along ('x', 'y', or 'z').
        width (int): Output image width.
        height (int): Output image height.
        background_color (tuple): RGB float values in [0, 1].
        output_path (str): Path to save the PNG file.
        text (str): Optional text to draw on the image.
        padding (int): Padding (in pixels) around the point cloud projection.
        xy_range (tuple): ((xmin, xmax), (ymin, ymax)) in world units to lock the viewing window.

    Returns:
        None
    """
    print("Rasterizing Cloud...")
    axis_map = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
    idx = axis_map[axis.lower()]
    points = np.asarray(pcd.points)

    # Prepare canvas
    bg_color = (np.array(background_color) * 255).astype(np.uint8)
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    if points.size == 0:
        print("Warning: No points to rasterize.")
        cv2.imwrite(output_path, img)
        return

    # Extract relevant 2D projection
    xy = points[:, idx]

    # Set bounds
    if xy_range is not None:
        (xmin, xmax), (ymin, ymax) = xy_range
    else:
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
    print(f"XRANGE: [{xmin},{xmax}]\tYRANGE:[{ymin},{ymax}]")

    span_x = xmax - xmin
    span_y = ymax - ymin
    drawable_width = width - 2 * padding
    drawable_height = height - 2 * padding

    if drawable_width <= 0 or drawable_height <= 0:
        raise ValueError("Padding too large for image size.")

    # Uniform scale
    scale = min(drawable_width / span_x, drawable_height / span_y) if span_x > 0 and span_y > 0 else 1.0

    # Transform to pixel coordinates
    scaled_xy = (xy - np.array([xmin, ymin])) * scale
    bbox_width, bbox_height = span_x * scale, span_y * scale
    offset = np.array([(width - bbox_width) / 2, (height - bbox_height) / 2])
    pixel_xy = scaled_xy + offset

    # Convert to int and flip Y axis
    pixel_xy = pixel_xy.astype(np.int32)
    pixel_xy[:, 1] = height - 1 - pixel_xy[:, 1]
    pixel_xy = np.clip(pixel_xy, [0, 0], [width - 1, height - 1])

    # Draw points
    for x, y in pixel_xy:
        img[y, x] = (0, 0, 0)

    # Add text
    if text:
        cv2.putText(img, text, (10, height - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Image saved to: {output_path}")


def find_files_by_pattern(root_dir,search_pattern="*align.ply"):
    """
    Recursively search for files matching '*align.ply' using glob.

    Args:
        root_dir (str): Directory to search.

    Returns:
        list of str: Full file paths to matching files.
    """
    pattern = os.path.join(root_dir, "**", search_pattern)
    return glob(pattern, recursive=True)

def get_info_text(path,position,thickness,label):
    parts = path.split('-')
    s = ""
    if parts[0] == "p2":
        s += "[Preliminaries] "
    elif parts[0] == "s":
        s += "[Semifinals] "
    elif parts[0] == "f":
        s += "[Finals] "
    elif parts[0] == "gt":
        s += "[Ground Truth] "
    elif parts[0] == "lab":
        s += "[SkyeBrowse] "

    s += parts[1]    
    # s += f" - Map {parts[2][-1]} "
    s += f"[Sliced at {position:.1f}m +/- {thickness/2:.1f}][{label}]"
    return s

# Example usage:
if __name__ == "__main__":
    # Load batch of cloud files by regex pattern:
    # input_dir = "./data/robocup2025/"
    # search_pattern = "lab*aligned-icp.ply"
    # cloud_paths = sorted(find_files_by_pattern(input_dir,search_pattern))
    # print("Found the following point cloud files:")
    # for path in cloud_paths:
    #     print(" > ", os.path.basename(path))
    # print(f"Total Files: {len(cloud_paths)}")
    # input("\nPress Enter to begin processing... (Ctrl+C to Cancel)")

    # Load single cloud to slice by file selection:
    cloud_paths = [select_point_cloud_files(".")]

    output_dir = "./data/slice"
    os.makedirs(output_dir, exist_ok=True)


    # Process all gathered files
    for fpath in cloud_paths:
        fname = os.path.basename(fpath)
        print(f"Reading Cloud... ( {fpath} )")
        pcd = o3d.io.read_point_cloud(fpath)
        # Get Low Slice
        position = 1.0
        thickness = 0.5
        sliced = slice_pointcloud(pcd, position=position, thickness=thickness, axis='z')
        out_path = os.path.join(output_dir, fname.replace(".ply", "-lowslice.png").replace(".pcd", "-lowslice.png"))
        
        try:
            info_text = get_info_text(fname,position,thickness,"LOW")
        except:
            print(f"[!] FAILED FOR {fname}")
            continue

        rasterize_pointcloud(sliced,
                            axis='z',
                            width=1000,
                            height=800,
                            background_color=(1.0, 1.0, 1.0),
                            output_path=out_path,
                            text=info_text,
                            xy_range=None) # optionally specify XY range
                            # xy_range=[(-1.0,12.0),(-4.0,3.5)])
        
        # Get High Slice
        position = 1.7
        thickness = 0.2
        sliced = slice_pointcloud(pcd, position=position, thickness=thickness, axis='z')
        out_path = os.path.join(output_dir, fname.replace(".ply", "-highslice.png").replace(".pcd", "-highslice.png"))
        
        try:
            info_text = get_info_text(fname,position,thickness,"HIGH")
        except:
            print(f"[!] FAILED FOR {fname}")
            continue

        rasterize_pointcloud(sliced,
                            axis='z',
                            width=1000,
                            height=800,
                            background_color=(1.0, 1.0, 1.0),
                            output_path=out_path,
                            text=info_text,
                            xy_range=None) # optionally specify XY range
                            # xy_range=[(-1.0,12.0),(-4.0,3.5)]) # finals crop
 
 