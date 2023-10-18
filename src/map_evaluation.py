#!/usr/bin/env python3
import sys
import os
import datetime
import csv
import cv2 as cv
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import point_matching as pm

position_changed = False
position = [0,0]
start_position = []
scale = 1


class eval_image:
    def __init__(self,dir,im_name):
        self.dir = dir
        self.im_file = im_name
        self.im_path = os.path.join(dir,im_name)
        self.image = cv.imread(self.im_path)
        self.point_dict = dict()

    def add_point(self,index,coords):
        self.point_dict[index] = coords
        print("INDEX " + str(index) + " SET: "+str(coords))
        print("New point dict")
        print(self.point_dict)

    def remove_point(self,index):
        try:
            self.point_dict.pop(index)
        except:
            print("Couldn't remove "+str(index))

    def export(self,save_dir=None):
        if save_dir is None:
            save_dir = self.dir
        filename = self.im_file[:-4]+".pts"
        f = open(os.path.join(save_dir,filename),"w+")
        f.write(self.im_file[:-4]+"\n")
        for p in sorted(self.point_dict.keys()):
            write_str = str(p)+","+str(self.point_dict[p][0])+","+str(self.point_dict[p][1])+"\n"
            f.write(write_str)
        f.close()
        print("POINT DICT EXPORTED TO " + filename)

    def import_pts(self,pts_file):
        print("Importing "+str(pts_file)+" ...")
        try:
            f = open(pts_file,"r")
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                data = line.split(',')
                # print(data)
                marker = int(data[0])
                x = int(data[1])
                y = int(data[2][:-1])
                coords = [x,y]
                self.point_dict[marker] = coords
            print("Imported point dict:")
            print(self.point_dict)
        except:
            print("PTS IMPORT FAILED")


def set_scale(pix,mm):
    '''Set scaling factor by providing pixels and the equivalent distance in millimeters'''
    global scale
    print("\nSetting scale")
    print(str(pix)+"px == "+str(mm)+"mm")
    scale = float(mm) / float(pix)
    print("Scaling factor = "+str(scale))
    return scale


def calculateDistance(x1,y1,x2,y2,precision=1):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dist = round(dist,precision)
    return dist


def click_event(event,x,y,flags,param):
    global position, position_changed, start_position
    
    if event == cv.EVENT_LBUTTONDOWN:
        start_position = [x, y]
    if event == cv.EVENT_LBUTTONUP:
        if (start_position == []):
            position = [x, y]
        else:
            position = [(start_position[0]+x)//2,(start_position[1]+y)//2]
            position_changed = True


def compose(image1, image2, blend=50):
    alpha = float(blend) / 100
    beta = 1.0 - alpha

    blended = cv.addWeighted(image1, alpha, image2, beta, 0.0)

    return blended


def draw_pts(image,pts,circle=False,label=True):
    '''Draw labelled marker points onto image'''
    im = image.copy()
    counter = 1
    for p in pts:
        center = (pts[p][0], pts[p][1])
        # cv.circle(im, center, 3, (0, 25, 255), 2)
        d = 5
        cv.line(im,(center[0],center[1]-d),(center[0],center[1]+d),(0,0,255),1)
        cv.line(im,(center[0]-d,center[1]),(center[0]+d,center[1]),(0,0,255),1)
        if circle:
            radius = p[2]
            cv.circle(im, center, radius, (255, 0, 100), 2)
        if label:
            font = cv.FONT_HERSHEY_PLAIN
            org = (pts[p][0],pts[p][1]-3)
            im = cv.putText(im,str(p),org,font,1,(255,255,255),2  ,cv.LINE_AA)
            im = cv.putText(im,str(p),org,font,1,(0,0,0),1  ,cv.LINE_AA)
        counter = counter + 1
    return im


def draw_textline(image,string,org=[5,20],scale=1.0,c=(0,0,0),c2=(255,255,255),t=2,border=True):
    font = cv.FONT_HERSHEY_PLAIN
    if border: image = cv.putText(image,string,org,font,scale,c2,t+1,cv.LINE_AA)
    image = cv.putText(image,string,org,font,scale,c,t,cv.LINE_AA)
    # print("Added text: \""+string+"\"")


def draw_resultsoverlay(image, data_dict, org=[5,0],scale=1.5,c=(0,0,0),c2=(255,255,255),t=2):
    im = image.copy()
    
    y_off = int(20 * scale)
    origin = org.copy()
    origin[1] = origin[1] + y_off
    
    if 'name' in data_dict:
        s = "Name: "+data_dict['name']
        if 'auto' in data_dict:
            s = s + "  [Auto-Adjusted]"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if 'scale' in data_dict:
        s = "Reference Scale: "+str(round(data_dict['scale'],2))+"mm/px"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if 'error' in data_dict:
        s = "Global Error: "+str(data_dict['error'])+"px"
        if 'scale' in data_dict:
            scaled = round(data_dict['error']*data_dict['scale'],2)
            s = s +" // "+str(scaled)+"mm"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off
    
    if 'std' in data_dict:
        s = "Std Dev: "+str(data_dict['std'])+"px"
        if 'scale' in data_dict:
            scaled = round(data_dict['std']*data_dict['scale'],2)
            s = s +" // "+str(scaled)+"mm"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if 'coverage' in data_dict:
        cover = round(data_dict['coverage'],1)
        s = "Coverage: "+str(cover)+"%"
        draw_textline(im,s,origin,scale,c,c2,t)

    return im
    

def export_reference_crops(dir,eval_image,crop_size=75,all_points=False):
    print("Exporting Reference Marker Crops...")
    save_dir = os.path.join(dir,"crops")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # print("Directory "+save_dir+" does not exist. Crops not saved.")
        # return
    d = crop_size//2
    img = eval_image.image
    marker_dict = eval_image.point_dict
    if all_points:
        for p in sorted(marker_dict.keys()):
            x = marker_dict[p][0]
            y = marker_dict[p][1]
            # print("x",x,"y",y)
            crop = img[y-d:y+d,x-d:x+d,:]
            crop_file = "marker_"+str(p)+".jpg"
            save_path = os.path.join(save_dir,crop_file)
            # print("save",save_path)
            cv.imwrite(save_path,crop)
    else: # Special fiducial mode
        pos = 1
        for i in range(1,11):
            fd = d
            a_exists = pos in marker_dict.keys()
            b_exists = pos+1 in marker_dict.keys()
            if a_exists and b_exists:
                x = (marker_dict[pos][0] + marker_dict[pos+1][0])//2
                y = (marker_dict[pos][1] + marker_dict[pos+1][1])//2
                dist = calculateDistance(marker_dict[pos][0],marker_dict[pos][1],marker_dict[pos+1][0],marker_dict[pos+1][1])
                if dist//2 > d: 
                    print("FIDUCIAL "+str(i)+" SEPARATED")
                    fd = int((dist//2) + d)
            elif a_exists:
                x = marker_dict[pos][0]
                y = marker_dict[pos][1]
            elif b_exists:
                x = marker_dict[pos+1][0]
                y = marker_dict[pos+1][1]
            else:
                continue
            crop = img[y-fd:y+fd,x-fd:x+fd,:]
            crop_file = "fiducial_"+str(i)+".jpg"
            save_path = os.path.join(save_dir,crop_file)
            # print("save",save_path)
            cv.imwrite(save_path,crop)
            # print(i,pos,a_exists,b_exists)
            pos = pos + 2
            
    
def export_log_data(dir,data,log_name="data_log.csv",headers=None):
    if headers is None:
        headers = ["Date","Test","Coverage","Error (px)","Std Dev (px)","Scale (mm/px)","Error (mm)","Std Dev (mm)","Mode"]
    log_file = os.path.join(dir,log_name)
    log_exists = os.path.exists(log_file)

    # Parse Data Dict
    data_list = []
    now = datetime.datetime.now()
    data_list.append(now.strftime('%Y-%m-%d'))
    data_list.append(data['name']) if 'name' in data else data_list.append("")
    data_list.append(data['coverage']) if 'coverage' in data else data_list.append("")
    data_list.append(data['error']) if 'error' in data else data_list.append("")
    data_list.append(data['std']) if 'std' in data else data_list.append("")
    if 'scale' in data: 
        data_list.append(data['scale'])
        scaled = round(data['error']*data['scale'],2)
        data_list.append(scaled)
        scaled = round(data['std']*data['scale'],2)
        data_list.append(scaled)
    else:
        data_list.append("")
        data_list.append("")
        data_list.append("")
    if 'auto' in data:
        data_list.append("Auto-Adjusted")
    else:
        data_list.append("Unadjusted")
    print(data_list)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists: 
            writer.writerow(headers)
        writer.writerow(data_list)


def main(argv):
    print("ARGV: {}".format(argv))
    dir = 'C:\\Users\\csrobot\\Documents\\decisive_map_eval'
    default_file = 'omron_map4.jpg'
    ref_file = argv[0] if len(argv) > 0 else default_file
    # eval_dir = 'E:\\Computing\\Projects\\map_testing\\aligned'
    eval_dir = 'C:\\Users\\csrobot\\Documents\\decisive_map_eval\\aligned'
    eval_files = [f for f in listdir(eval_dir) if isfile(join(eval_dir, f))] 
    # eval_files = ['auto_align_test.jpg']
    eval_pos = 0
    trans_file = eval_files[eval_pos]

    # Loads an image
    ref_image = eval_image(dir,ref_file)
    trans_image = eval_image(eval_dir,trans_file)
    image_list = [ref_image, trans_image]

    # Check if image is loaded fine
    if ref_image.image is None:
        print ('Error opening image!')
        return -1
    else:
        print("Loaded {} - {}".format(ref_file,ref_image.image.shape[:2]))
        print("Loaded {} - {}".format(trans_file,trans_image.image.shape[:2]))

    #CV Window and Trackbars
    cv.namedWindow("Map Comparison",cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("Map Comparison", click_event)

    # SET SCALE - PIXELS to equivalent MM
    scale = set_scale(27,610)

    image_selection = 0
    marker_position = 1
    running = True
    global position_changed
    while running:

        if position_changed:
            image_list[image_selection].add_point(marker_position,position)
            position_changed = False

        ref = ref_image.image.copy()
        trans = trans_image.image.copy()

        ref = draw_pts(ref,ref_image.point_dict)
        im2 = draw_pts(trans,trans_image.point_dict)
        image_list2 = [ref, im2]
        # comp = compose(ref,im2)

        # KEYBOARD CONTROLS
        k = cv.waitKey(100) & 0xFF
        if k == ord('1'): # View reference image
            image_selection = 0
        if k == ord('2'): # View eval image
            image_selection = 1
        if k == ord('3'): # Decrease marker position
            marker_position = max(0,marker_position - 1)
            print("POS: "+str(marker_position))
        if k == ord('4'): # Increase marker position
            marker_position = marker_position + 1
            print("POS: "+str(marker_position))
        if k == ord('5'): # Set marker position to 1
            marker_position = 1
            print("POS: "+str(marker_position))
        if k == ord('6'): # Prev eval image
            eval_pos = (eval_pos - 1) % len(eval_files)
            trans_file = eval_files[eval_pos]
            trans_image = eval_image(eval_dir,trans_file)
            image_list = [ref_image, trans_image]
        if k == ord('7'): # Next eval image
            eval_pos = (eval_pos + 1) % len(eval_files)
            trans_file = eval_files[eval_pos]
            trans_image = eval_image(eval_dir,trans_file)
            image_list = [ref_image, trans_image]
            print("LOADED: "+trans_file)
        if k == ord('w'): # Remove selected reference point from image
            image_list[image_selection].remove_point(marker_position)
        if k == ord('e'): # Evaluate map without adjustment
            composed_img = compose(ref_image.image,trans_image.image,50)
            ref_dict, test_dict, coverage = pm.calc_coverage(ref_image.point_dict, trans_image.point_dict)
            err_px, std_px, err_img = pm.calc_error(ref_dict, test_dict, composed_img)
            # Create result dictionary
            scores = dict()
            scores['name'] = image_list[image_selection].im_file[:-4]
            scores['scale'] = scale
            scores['error'] = err_px
            scores['std'] = std_px
            scores['coverage'] = coverage
            score_img = draw_resultsoverlay(err_img,scores)
            # Save Results
            result_dir = os.path.join(dir,"results")
            if not os.path.exists(result_dir):
                # os.makedirs(result_dir)
                print("Location "+result_dir+" does not exist. Result not saved.")
            else:
                # Create Test Subfolder
                test_dir = result_dir
                # test_dir = os.path.join(result_dir,scores['name'])
                if not os.path.exists(test_dir): 
                    os.makedirs(test_dir)
                # Save Result Image
                result_file = scores['name']+"_result.jpg"
                cv.imwrite(os.path.join(test_dir,result_file),score_img)
                print("Saved result image "+result_file+".")
                # Save Marker Crop Images
                # export_reference_crops(test_dir,image_list[image_selection],crop_size=75)
                # Log Result Data
                export_log_data(result_dir, scores)
            # Show Result Image
            cv.imshow("Map Comparison", score_img)
            cv.waitKey()
        if k == ord('r'): # Evaluate map with auto-adjustment
            viz = False # Visualize autoscaling process?
            test_img, test_dict = pm.auto_fullprocess(ref_image.point_dict, trans_image.point_dict, trans_image.image,verbose=True,visualize=viz)
            composed_img = compose(ref_image.image,test_img,30)
            if viz:
                cv.imshow("Map Comparison", composed_img)
                cv.waitKey()
            ref_dict, test_dict, coverage = pm.calc_coverage(ref_image.point_dict, test_dict)
            err_px, std_px, err_img = pm.calc_error(ref_dict, test_dict, composed_img)
            # Create result dictionary
            scores = dict()
            scores['name'] = image_list[image_selection].im_file[:-4]
            scores['scale'] = scale
            scores['error'] = err_px
            scores['std'] = std_px
            scores['coverage'] = coverage
            scores['auto'] = True
            score_img = draw_resultsoverlay(err_img,scores)
            # Save Results
            result_dir = os.path.join(dir,"results")
            if not os.path.exists(result_dir):
                # os.makedirs(result_dir)
                print("Location "+result_dir+" does not exist. Result not saved.")
            else:
                # Create Test Subfolder
                test_dir = result_dir
                # test_dir = os.path.join(result_dir,scores['name'])
                if not os.path.exists(test_dir): 
                    os.makedirs(test_dir)
                # Save Result Image
                result_file = scores['name']+"_auto-result.jpg"
                cv.imwrite(os.path.join(test_dir,result_file),score_img)
                print("Saved result image "+result_file+".")
                # Save Marker Crop Images
                # export_reference_crops(test_dir,image_list[image_selection],crop_size=75)
                # Log Result Data
                export_log_data(result_dir, scores)
            # Show Result Image
            cv.imshow("Map Comparison", score_img)
            cv.waitKey()

        if k == ord('x'): # Export reference .pts file as along with an image displaying them
            filename = "pts\\"+image_list[image_selection].im_file[:-4]+"_pts.jpg"
            image_list[image_selection].export(os.path.join(dir,"pts"))
            cv.imwrite(os.path.join(dir,filename),image_list2[image_selection])
        if k == ord('z'): # Import .pts file for the currently selected image
            pts_file = "pts\\"+image_list[image_selection].im_file[:-4]+".pts"
            pts_file = os.path.join(dir,pts_file)
            image_list[image_selection].import_pts(pts_file)

        if k == ord('q'): # Exit program
            running = False

        img_text = "IMG:"+str(image_selection+1)+" MARKER:"+str(marker_position)+" IMG: "+image_list[image_selection].im_file
        font = cv.FONT_HERSHEY_PLAIN
        org = (5,25)
        image_list2[image_selection] = cv.putText(image_list2[image_selection],img_text,org,font,2,(0,0,0),2,cv.LINE_AA)

        cv.imshow("Map Comparison", image_list2[image_selection])

    return 0
if __name__ == "__main__":
    main(sys.argv[1:])
