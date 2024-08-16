import os
import cv2
import tkinter as tk
from tkinter import filedialog
import argparse
import point_manip as pm 
import utilities as util
import csv_loader

mouse_x, mouse_y = 0, 0
click_flag = False
eval_pt_scale = 1
eval_pt_rotation = 0 

# Callback function for the trackbars
def update_scale(val):
    print(f"Scale: {val}")

def update_rotation(val):
    print(f"Rotation: {val}")

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, click_flag
    if event == cv2.EVENT_LBUTTONUP:
        mouse_x, mouse_y = x, y
        click_flag = True
        print(f"Mouse released at ({mouse_x}, {mouse_y})")

    
def main(args):
    global mouse_x, mouse_y, click_flag, eval_pt_rotation, eval_pt_scale
    flip_x = args.flip_x
    flip_y = args.flip_y

    # Load Map Image
    print(f"Loading map image: {args.img_file}")
    map_img = cv2.imread(args.img_file)
    height, width, _ = map_img.shape
    map_img_center = [width//2, height//2]

    # Read in points file
    # NOTE: Default arguments assume headers are on the first line of the file
    print(f"Loading points file: {args.pts_file}")
    headers, data = csv_loader.read_csv_points(args.pts_file)
    data = csv_loader.fix_data_types(data,set_str=['label'],set_float=['x','y','z'])
    data = pm.flip_points(data, flip_x, flip_y)
    # for point in data:
    #     print(point)
    # eval_pts = util.import_pts(args.pts_file)
    data = pm.move_points_to(data,map_img_center)
    # init_scale = 100 
    # eval_pts = pm.scale_dict(eval_pts, init_scale)
    data_copy = data.copy()

    # CV Window
    cv2.namedWindow('Map Align')
    cv2.createTrackbar('Scale', 'Map Align', int(args.scale), 1000, update_scale)  # scale from 0.0 to 10.0
    cv2.createTrackbar('Rotation', 'Map Align', int(args.rotation), 360, update_rotation)  # rotation from 0 to 360 degrees
    cv2.setMouseCallback('Map Align', mouse_callback)


    running = True
    update = True
    while running:
        if click_flag:
            data = pm.move_points_to(data,[mouse_x,mouse_y])
            data_copy = pm.move_points_to(data_copy,[mouse_x,mouse_y])
            update = True
            click_flag = False
        
        tb_scale = cv2.getTrackbarPos('Scale', 'Map Align')
        if tb_scale != eval_pt_scale:
            eval_pt_scale = tb_scale
            # mp = pm.get_dict_midpoint(data_copy)
            data_copy = data.copy()
            data_copy = pm.scale_points(data_copy,eval_pt_scale,verbose=False)
            data_copy = pm.rotate_points(data_copy,eval_pt_rotation)
            # data_copy = pm.move_dict_to(data_copy,mp)
            update = True
        tb_rot = cv2.getTrackbarPos('Rotation', 'Map Align')
        if tb_rot != eval_pt_rotation:
            print(f"Dif:{eval_pt_rotation-tb_rot}")
            data_copy = pm.rotate_points(data_copy,tb_rot-eval_pt_rotation)
            eval_pt_rotation = tb_rot
            update = True

        if update:
            display_img = map_img.copy()

            # Draw Points 
            for p in data_copy: 
                util.draw_ref_pt(display_img,[p['x'],p['y']],p['label'],color=(255,0,255))

            update = False
        # Display the image 
        cv2.imshow("Map Align",display_img)

        #Keyboard Controls
        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            running = False
        if k == ord('e'): # Export
            suffix = "_pxaligned"
            export_file = os.path.basename(args.output_dir).split('.')[0] + suffix + ".csv"
            export_path = os.path.join(args.output_dir,export_file)
            csv_loader.write_csv(export_path, headers, data_copy)
            update = True

if __name__ == "__main__":
    img_file = None
    img_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_gt.png"
    pts_file = None
    # pts_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_out.csv"
    output_dir = "../data/example"
    output_dir = None
    initial_scale = 100                             # Initial Scale (100 = m -> cm)
    initial_rotation = 90                           # Initial Rotation
    flip_x = True
    flip_y = False

    parser = argparse.ArgumentParser(description="Tool for aligning points to a map image.")
    parser.add_argument('-i', '--img_file', dest='img_file', type=str, default=img_file,
                        help='Directory for saving output images')
    parser.add_argument('-p', '--pts_file', dest='pts_file', type=str, default=pts_file,
                        help='Directory for saving output images')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default=output_dir,
                        help='Directory for saving output images')
    parser.add_argument('--scale', dest='scale', type=int, default=initial_scale,
                        help='Initial scale to apply to points')
    parser.add_argument('--rotation', dest='rotation', type=int, default=initial_rotation,
                        help='Initial rotation to apply to points')
    parser.add_argument('--flip-x', dest='flip_x', type=bool, default=flip_x,
                        help='Whether to flip the x values of eval points')
    parser.add_argument('--flip-y', dest='flip_y', type=bool, default=flip_y,
                        help='Whether to flip the y values of eval points')
    
    args = parser.parse_args()

    if args.img_file is None:
        # Create a simple GUI to select a file
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.geometry(f"{1920}x{1080}+{0}+{0}")
        # Ask the user to select an image file using a file dialog
        print("No image file defined in script or arguments, please select one.")
        initial_dir = '.'
        img_file = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.pgm *.ppm *.pnm")], initialdir=initial_dir)
        # file_name = os.path.basename(file_path)
        # file_dir = os.path.dirname(file_path)
        root.destroy()
        args.img_file = img_file

    if args.pts_file is None:
        args.pts_file = csv_loader.open_csv_dialog()

    # Output directory defaults to where the points file is
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pts_file)

    main(args)