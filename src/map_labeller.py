#!/usr/bin/env python3
import os
from os import listdir
from os.path import isfile, join
import cv2
import tkinter as tk
from tkinter import filedialog
import argparse
import utilities as util
import csv_loader

position = None
position_changed = False
lbl_root = None

def click_event(event,x,y,flags,param):
    global position, position_changed, start_position
    # start_position = []
    if event == cv2.EVENT_LBUTTONDOWN:
        start_position = [x, y]
    if event == cv2.EVENT_LBUTTONUP:
        if (start_position == []):
            position = [x, y]
        else:
            position = [(start_position[0]+x)//2,(start_position[1]+y)//2]
        position_changed = True

def center_window(win, width, height):
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    win.geometry(f"{width}x{height}+{x}+{y}")

def on_enter(event=None):
    global text,lbl_root,popup_window
    text = entry.get()
    print("Label Named:", text)
    popup_window.destroy()

def show_popup(default_text=""):
    global popup_window,entry,lbl_root
    if lbl_root is None:
        lbl_root = tk.Tk()
        lbl_root.withdraw()  # Hide the main window

    popup_window = tk.Toplevel(takefocus=True)
    popup_window.title("Enter Reference Point Label")
    popup_window.focus_force()
    entry = tk.Entry(popup_window)
    entry.insert(0, default_text)  # Default text in the entry widget
    entry.select_range(0, 'end')     # Highlight all text
    entry.focus_set()                # Set focus to the entry widget
    entry.focus_force()
    popup_window.bind('<Return>', on_enter)

    enter_button = tk.Button(popup_window, text="Enter", command=on_enter)

    entry.pack(padx=10, pady=10)
    enter_button.pack(pady=10)
    center_window(popup_window,200,100)
    # while not submitted:
    lbl_root.wait_window(popup_window)
    return text

def main(args):
    global position_changed, position
    running = True
    update = True

    # Load points file if given
    if args.pts_file is not None:
        headers, data = csv_loader.read_csv_points(csv_path)
        data = csv_loader.fix_data_types(data,set_str=['label'],set_float=['x','y'])
    
    # Load the selected image using OpenCV
    image = cv2.imread(args.img_file)
    draw_img = image.copy()

    #CV Window
    cv2.namedWindow("Map Image",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Map Image", click_event)
    # cv2.setWindowProperty("Map Image", cv2.WND_PROP_TOPMOST, 1)
    if image is None:
        print("Failed to load the image.")
        running = False

    data = []
    pt_label = None
    current_index = 0
    while running:
        # True when map image is clicked: move or create point
        if position_changed:
            if pt_label is None:
                pt_label = show_popup(str(len(data)+1))
                data.append(dict())

            print("Label: {}     Pos: {}".format(pt_label,position))
            data[current_index]['label'] = pt_label
            data[current_index]['x'] = position[0]
            data[current_index]['y'] = position[1]
            # Redraw map image
            update = True
            position_changed = False

        # True when image should be redrawn
        if update:
            # Copy original map image
            draw_img = image.copy()

            # Draw all reference points in data
            for p in data: 
                util.draw_ref_pt(draw_img,[p['x'],p['y']],p['label'],color=(0,0,255))

            #Draw Status Text:
            if len(data) <= current_index:
                current_key = None
                current_val = None
            else:
                current_key = data[current_index]['label']
                current_val = [data[current_index]['x'],data[current_index]['y']]
            status_str = "{}/{} - {}: {}".format(current_index+1,len(data),current_key,current_val)
            util.draw_textline(draw_img,status_str,scale = 2)
            update = False

        # Display the image 
        cv2.imshow("Map Image",draw_img)

        #Keyboard Controls ==============================================================
        k = cv2.waitKey(100) & 0xFF
        # QUIT PROGRAM
        if k == ord('q'):
            running = False
        # RENAME CURRENT POINT
        if k == ord('r'):
            if len(data) <= current_index:
                print("Nothing to rename.")
                
            else:
                data[current_index]['label'] = show_popup(str(data[current_index]['label']))
                update = True
        # SAVE IMAGE
        if k == ord('s'): 
            util.save_image_dialog(draw_img,args.output_dir)
        # IMPORT PTS CSV FILE
        if k == ord('z'): 
            csv_path = csv_loader.open_csv_dialog(args.output_dir)
            if csv_path is not None:
                headers, data = csv_loader.read_csv_points(csv_path)
                data = csv_loader.fix_data_types(data,set_str=['label'],set_float=['x','y'])
            update = True
        # EXPORT PTS CSV FILE
        if k == ord('x'): 
            csv_path = csv_loader.save_csv_dialog(data,args.output_dir)
            im_file = os.path.join(os.path.dirname(csv_path),os.path.basename(csv_path)[:-4]+'.png')
            print(f"IM FILE: {im_file}")
            cv2.imwrite(im_file,draw_img)
            print(f"Saved {im_file}")
        # PREVIOUS REF POINT
        if k == ord('1'): 
            current_index = (current_index-1)%(len(data)+1)
            if len(data) <= current_index:
                pt_label = None
            else:
                pt_label = data[current_index]['label']
            update = True
            print(f"Index [{current_index}] - Label: {pt_label}")
        # NEXT REF POINT
        if k == ord('2'): 
            current_index = (current_index+1)%(len(data)+1)
            if len(data) <= current_index:
                pt_label = None
            else:
                pt_label = data[current_index]['label']
            update = True
            print(f"Index [{current_index}] - Label: {pt_label}")
        # REMOVE REF POINT
        if k == ord('3'): 
            if len(data) <= current_index:
                pass
                print("Nothing removed.")
            else:
                popped = data.pop(current_index)
                print(f"Removed Index {current_index}: {popped}")
                update = True

if __name__ == "__main__":
    # Define default parameter values
    img_file = None
    # img_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_gt.png"
    
    pts_file = None

    output_dir = None
    output_dir = "../data/example"

    parser = argparse.ArgumentParser(description="Tool for aligning points to a map image.")
    parser.add_argument('-i', '--img_file', dest='img_file', type=str, default=img_file,
                        help='Directory for saving output images')
    parser.add_argument('-p', '--pts_file', dest='pts_file', type=str, default=pts_file,
                        help='Directory for saving output images')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default=output_dir,
                        help='Directory for saving output images')
    args = parser.parse_args()

    # Open dialog to select map image if no path is given
    if args.img_file is None:
        img_file = util.open_image_dialog(args.output_dir)
        args.img_file = img_file

    # Select output directory if none is given
    if args.output_dir is None:
        args.output_dir = util.select_directory()
        if args.output_dir is None:
            args.output_dir = "."

    main(args)