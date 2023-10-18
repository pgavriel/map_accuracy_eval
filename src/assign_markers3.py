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

search_rad = 50
position_changed = False
position = [0,0]
start_position = []
blend = 50
ref_pts = []
im2_pts = []
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
                if data[0].isnumeric():
                    marker = int(data[0])
                else:
                    marker = data[0]
                x = int(data[1])
                y = int(data[2][:-1])
                coords = [x,y]
                self.point_dict[marker] = coords
            print("Imported point dict:")
            print(self.point_dict)
        except:
            print("PTS IMPORT FAILED")



def set_blend(v):
    global blend
    blend = v

def set_scale(pix,mm):
    # Set scaling factor by providing pixels and the equivalent distance in millimeters
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
    # start_position = []
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

# Draw labeled marker points
def draw_pts(image,pts,circle=False,label=True):
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

def draw_textline(image,string,org=[5,0],scale=1.0,c=(0,0,0),c2=(255,255,255),t=2):
    font = cv.FONT_HERSHEY_PLAIN
    image = cv.putText(image,string,org,font,scale,c2,t+1,cv.LINE_AA)
    image = cv.putText(image,string,org,font,scale,c,t,cv.LINE_AA)
    print("Added text: \""+string+"\"")

def draw_resultsoverlay(image, data_dict, org=[5,0],scale=1.5,c=(0,0,0),c2=(255,255,255),t=2):
    im = image.copy()
    
    y_off = int(20 * scale)
    origin = org.copy()
    origin[1] = origin[1] + y_off
    
    if data_dict['name']:
        s = "Name: "+data_dict['name']
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if data_dict['scale']:
        s = "Reference Scale: "+str(round(data_dict['scale'],2))+"mm/px"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if data_dict['error']:
        s = "Global Error: "+str(data_dict['error'])+"px"
        if data_dict['scale']:
            scaled = round(data_dict['error']*data_dict['scale'],2)
            s = s +" // "+str(scaled)+"mm"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off
    
    if data_dict['std']:
        s = "Std Dev: "+str(data_dict['std'])+"px"
        if data_dict['scale']:
            scaled = round(data_dict['std']*data_dict['scale'],2)
            s = s +" // "+str(scaled)+"mm"
        draw_textline(im,s,origin,scale,c,c2,t)
        origin[1] = origin[1] + y_off

    if data_dict['coverage']:
        cover = round(data_dict['coverage'],1)
        s = "Coverage: "+str(cover)+"%"
        draw_textline(im,s,origin,scale,c,c2,t)

    return im
    


def export_reference_crops(dir,eval_image,crop_size=75,all_points=False):
    print("Exporting Reference Marker Crops...")
    save_dir = os.path.join(dir,"crops")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
        headers = ["Date","Test","Coverage","Error (px)","Std Dev (px)","Scale (mm/px)","Error (mm)","Std Dev (mm)"]
    log_file = os.path.join(dir,log_name)
    log_exists = os.path.exists(log_file)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists: 
            writer.writerow(headers)
        writer.writerow(data)



def calc_error(ref_dict,test_dict,ref_image,test_image,label="Test Not Specified"):
    print("CALCULATING ERROR")
    draw_img = compose(ref_image,test_image,20)
    dict1 = ref_dict.copy()
    dict2 = test_dict.copy()

    dlist1 = []
    dlist2 = []
    errors = []
    error_avg = []

    # Make sure both dicts only contain matching entries
    for p in ref_dict:
        if dict2.get(p) is not None:
            pass
            # print("Entry "+str(p)+" found in test dict")
        else:
            print("Entry "+str(p)+" NOT FOUND in TEST")
            dict1.pop(p)
    for p in test_dict:
        if dict1.get(p) is None:
            print("Entry "+str(p)+" NOT FOUND in REF")
            dict2.pop(p)

    # Should have identical keys
    # print(dict1)
    # print(dict2)
    if len(ref_dict) is not 0:
        coverage = (float(len(dict1)) / len(ref_dict)) * 100
    else:
        print("Ref dict empty")
        coverage = 100
    print("COVERAGE: "+str(coverage)+"%")
    
    for p in sorted(dict1.keys()):
        p1 = [dict1[p][0], dict1[p][1]]
        p2 = [dict2[p][0], dict2[p][1]]
        d1 = []
        d2 = []
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1]]
            p4 =[dict2[j][0], dict2[j][1]]
            if p1 == p3: continue

            cv.line(draw_img,tuple(p2),tuple(p4),(0,0,255),1)
            # cv.line(draw_img,tuple(p1),tuple(p3),(0,255,0),1)

            cv.line(test_image,tuple(p2),tuple(p4),(0,0,255),1)
            cv.line(ref_image,tuple(p1),tuple(p3),(0,255,0),1)

            d1.append(calculateDistance(p1[0],p1[1],p3[0],p3[1]))
            d2.append(calculateDistance(p2[0],p2[1],p4[0],p4[1]))
        e = np.asarray(d1) - np.asarray(d2)
        e = np.around(e,decimals=1)
        e = np.abs(e)
        # print(e)
        errors.append(list(e))
        e = round(float(np.average(e)),1)
        error_avg.append(e)
        dlist1.append(d1)
        dlist2.append(d2)

    # Overlay reference web
    for p in sorted(dict1.keys()):
        p1 = [dict1[p][0], dict1[p][1]]
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1]]
            cv.line(draw_img,tuple(p1),tuple(p3),(0,255,0),1)

    # print("\nDLIST1")
    # for i in dlist1: print(i)
    # print("\nDLIST2")
    # for i in dlist2: print(i)
    # print("---")
    # print("\nERRORS")
    # for i in errors: print(i)
    print("\nERROR_AVG",error_avg)
    #
    # print("Scaling ERROR_AVG ("+str(scale)+")")
    # print("SCALED ERROR AVG",error_avg*scale)

    std = round(np.std(error_avg),2)
    scaled_std = round(std*scale)
    score = round(np.average(np.asarray(error_avg)),2)
    print("SCORE",score)
    # print("Scaling score ("+str(scale)+")")
    scaled_score = score*scale
    print("SCALED SCORE",scaled_score,"mm\n")

    img_text = "Coverage: "+str(round(coverage))+"%"
    font = cv.FONT_HERSHEY_PLAIN
    org = (5,draw_img.shape[0]-70)
    img = cv.putText(draw_img,img_text,org,font,2,(255,255,255),3,cv.LINE_AA)
    img = cv.putText(draw_img,img_text,org,font,2,(0,0,0),2,cv.LINE_AA)

    img_text = "Global Error: "+str(score)+"px // "+str(round(scaled_score))+"mm  Std Dev: "+str(std)+"px // "+str(scaled_std)+"mm"
    org = (5,draw_img.shape[0]-45)
    img = cv.putText(draw_img,img_text,org,font,2,(255,255,255),3,cv.LINE_AA)
    img = cv.putText(draw_img,img_text,org,font,2,(0,0,0),2,cv.LINE_AA)

    org = (5,draw_img.shape[0]-20)
    img = cv.putText(draw_img,label,org,font,2,(255,255,255),3,cv.LINE_AA)
    img = cv.putText(draw_img,label,org,font,2,(0,0,0),2,cv.LINE_AA)

    #LOG DATA
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    data = [date_str, label, coverage, score, std, scale, scaled_score, scaled_std]
    # ref_data_str = ",,"
    # eval_data_str= str(coverage)+","+str(score)+","
    # for p in sorted(ref_dict.keys()):
    #     pass

    return draw_img, ref_image, test_image, data

# def get_dict_midpoint(test_dict):
#     test_midpoint = [0, 0]
#     for k in test_dict:
#         test_midpoint = [test_midpoint[0]+test_dict[k][0],test_midpoint[1]+test_dict[k][1]]
#     test_midpoint = [test_midpoint[0]//len(test_dict),test_midpoint[1]//len(test_dict)]
#     return test_midpoint

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_minerror_rotation(ref_dict, test_dict, mid_point,draw_img):
    step_size = 5
    min_step = step_size / 16
    position = 0
    d_sum = 0
    for k in test_dict:
        t_pt = test_dict[k]
        r_pt = ref_dict[k]
        d_sum = d_sum + calculateDistance(t_pt[0],t_pt[1],r_pt[0],r_pt[1])
        draw_img = cv.circle(draw_img, t_pt, 1, (0,0,255), 2)
    last_sum = d_sum
    min_sum = last_sum
    min_pos = position
    cval = 0
    print("{:2f} rotation d_sum: {:2f}".format(position,d_sum))
    while abs(step_size) >= min_step:
        position = position + step_size 
        cval = cval + 10
        d_sum = 0
        for k in test_dict:
            new_x,new_y = rotate(mid_point, test_dict[k], position)
            test_dict[k] = [new_x,new_y]
            t_pt = test_dict[k]
            r_pt = ref_dict[k]
            d_sum = d_sum + calculateDistance(t_pt[0],t_pt[1],r_pt[0],r_pt[1])
            draw_img = cv.circle(draw_img, [int(t_pt[0]),int(t_pt[1])], 1, (cval,cval,cval), 1)
        
        print("{:2f} rotation d_sum: {:2f}".format(position,d_sum))

        if d_sum > min_sum:
            step_size = -(step_size * 0.5)
            print("REFINE - NEW STEP ",step_size)
        else:
            min_sum = d_sum
            min_pos = position
        last_sum = d_sum
        
    print("MINIMUM ROTATION FOUND (CCW): ",min_pos)
    return min_pos, draw_img

    pass

def iterative_point_march(ref_dict,test_dict,draw_img):
    pass
    
    for k in test_dict:
        rp = ref_dict[k]
        tp = test_dict[k]
        dxy = [rp[0]-tp[0],rp[1]-tp[1]]


# def autoscaling_calc_error(ref_dict,test_dict,ref_image,test_image,label="Test",range=0.2,step=0.01):
#     print("Attempting Autoscaling Global Error")
#     print("Scale Factor Range (+/-):"+str(range))
#     print("Step Size: ",str(step))
#     test_img = test_image.copy()
#     current_scale = -range
#     count = 1
#     color_inc = 255 // int((range*2)/step)+1
#     print("COLOR INC",color_inc)
#     min_error = None
#     min_scale = None
#     scores_list = []
#     test_midpoint = get_dict_midpoint(test_dict)
#     ref_midpoint = get_dict_midpoint(ref_dict)
#     print("Ref MP:",ref_midpoint)
#     print("Test MP:",test_midpoint)
#     xy_off = [test_midpoint[0]-ref_midpoint[0],test_midpoint[1]-ref_midpoint[1]]
#     print("D MP:",xy_off)
#     for p in test_dict:
#         test_dict[p] = [test_dict[p][0]-xy_off[0],test_dict[p][1]-xy_off[1]]
#     test_midpoint = ref_midpoint
#     # return
#     # Find the correct scaling
#     while current_scale <= range:
#         cval = color_inc * count #Color value
#         scale_factor = round(1-current_scale,4)
#         print(str(count)+". Testing Scale:"+str(scale_factor))
#         # Scale Test Dict
#         scaled_dict = test_dict.copy()
#         for m in scaled_dict:
#             scaled_x = int((scaled_dict[m][0]-test_midpoint[0]) * scale_factor)+test_midpoint[0]
#             scaled_y = int((scaled_dict[m][1]-test_midpoint[1]) * scale_factor)+test_midpoint[1]
#             scaled_dict[m] = [scaled_x, scaled_y]
#             test_img = cv.circle(test_img, [scaled_x, scaled_y], 1, (0,255-cval,cval), 2)
#         # print("Scaled Dict:",str(scaled_dict))
#         _, _, _, data = calc_error(ref_dict,scaled_dict,ref_image,test_image,"Autoscaling-"+str(scale_factor*100))
#         score = data[3]
#         scores_list.append([scale_factor, score])
#         if min_error is None or score < min_error:
#             min_error = score
#             min_scale = scale_factor
#             min_scaled_dict = scaled_dict.copy()

#         current_scale = round(current_scale + step,4)
#         count = count + 1
#     test_img = cv.circle(test_img, test_midpoint, 3, (255,0,0), 2)

#     min_rotation, rot_img = get_minerror_rotation(test_dict,min_scaled_dict,test_midpoint,ref_image.copy())

#     for s in scores_list:
#         print("Scale: {}\tGlobal Error (px): {}".format(s[0],s[1]))
#     print("MINIMUM Scale: {}\tError: {}".format(min_scale,min_error))
#     cv.imshow("Autoscale test",rot_img)
#     cv.waitKey()
    #PRINT RESULTS


# def refine_marker(im, pts_list, min_circles=5):
#     global search_rad,position
#     print("\nFinding Circles at {}".format(position))
#     gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#     gray = cv.medianBlur(gray, 5)
#     rows = gray.shape[0]
#     circles = None
#     p1 = 200
#     p2 = 30
#     min_dist = 1
#     min_rad = 5
#     found_enough = False
#     while circles is not None or found_enough is False:
#         # print("P2: {}".format(p2))
#         circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, min_dist, param1=p1, param2=p2, minRadius=min_rad, maxRadius=search_rad)
#         if circles is not None:
#             print("Parameter 2: {} - {} Circles Found".format(p2,len(circles[0])))
#             if len(circles[0]) >= min_circles:
#                 found_enough = True
#                 break
#         else:
#             print("Parameter 2: {} - No Circles Found".format(p2))

#         p2 = p2 - 1
#         if p2 < 1:
#             circles = [[[search_rad,search_rad,search_rad]]]
#             break

#     print(circles)
#     print(circles[0])
#     for i in circles[0]:
#         center = (i[0], i[1])
#         # circle center
#         cv.circle(im, center, 1, (0, 100, 100), 1)
#         # circle outline
#         radius = i[2]
#         cv.circle(im, center, radius, (255, 0, 255), 1)

#     if len(circles[0]) > 1:
#         average_circle = sum(circles[0]) / len(circles[0])
#     else:
#         average_circle = circles[0][0]
#     print("Average Circle: {}".format(average_circle))
#     center = (average_circle[0], average_circle[1])
#     radius = average_circle[2]
#     cv.circle(im, center, 1, (0, 25, 25), 2)
#     cv.circle(im, center, radius, (255, 0, 100), 2)

#     avg = [int(average_circle[0]+position[0]-search_rad),int(average_circle[1]+position[1]-search_rad),int(average_circle[2])]

#     pts_list.append(avg)
#     return im


# def find_markers(im):
#     global search_rad,position,position_changed
#     global ref_pts, im2_pts
#     p = position
#     r = search_rad

#     crop = im.image[p[1]-r:p[1]+r,p[0]-r:p[0]+r].copy()

#     crop = refine_marker(crop,im.points_list)


#     cv.imshow("refimg",ref_crop)
#     cv.imshow("im2",crop)
#     position_changed = False

def main(argv):
    print("ARGV: {}".format(argv))
    dir = 'D:/map_accuracy_eval/example'
    default_file = 'spot_map_image.png'
    ref_file = argv[0] if len(argv) > 0 else default_file
    # eval_dir = 'E:\\Computing\\Projects\\map_testing\\aligned'
    eval_dir = 'D:/map_accuracy_eval/example'
    eval_files = [f for f in listdir(eval_dir) if isfile(join(eval_dir, f))] 
    eval_files = ['omron_map_new.png']
    eval_pos = 0
    trans_file = eval_files[eval_pos]
    # Loads an image
    ref_image = eval_image(dir,ref_file)
    trans_image = eval_image(eval_dir,trans_file)
    image_list = [ref_image, trans_image]
    blank = np.zeros(ref_image.image.shape,dtype=np.uint8)
    blank.fill(255)
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
    # cv.createTrackbar('blend', "Markers", 0, 100, set_blend)
    # cv.setTrackbarPos("blend", "Markers", blend)

    #SET SCALE
    # PIXELS to equivalent MM
    scale = set_scale(27,610)

    #Text settings
    txt_font = cv.FONT_HERSHEY_PLAIN

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
        # if k == ord('s'):
        #     cv.imwrite(os.path.join(dir,"adj_map_ref.jpg"),ref_image)
        #     cv.imwrite(os.path.join(dir,"adj_map_trans.jpg"),transform_image)
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
        if k == ord('e'): # Calculate and save error webs between reference image and selected image
            test_name = image_list[image_selection].im_file[:-4]
            result_web, ref_web, test_web, data = calc_error(ref_image.point_dict,trans_image.point_dict,ref_image.image,image_list2[image_selection],test_name)
            cv.imshow("Error Result",result_web)
            cv.waitKey(2)

            test_dir = os.path.join(dir,"results")
            test_dir = os.path.join(test_dir,test_name)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            # SAVE ERROR WEB IMAGES
            error_file = image_list[image_selection].im_file[:-4]+"_web.jpg"
            cv.imwrite(os.path.join(test_dir,error_file),result_web)
            print(error_file)
            error_file = image_list[image_selection].im_file[:-4]+"_ref.jpg"
            cv.imwrite(os.path.join(test_dir,error_file),ref_web)
            print(error_file)
            error_file = image_list[image_selection].im_file[:-4]+"_test.jpg"
            cv.imwrite(os.path.join(test_dir,error_file),test_web)
            print(error_file)
            # SAVE MARKER CROP IMAGES
            export_reference_crops(test_dir,image_list[image_selection],crop_size=75)
            # LOG DATA TO CSV
            export_log_data(os.path.join(dir,"results"), data)
        if k == ord('r'): # Autoscaling error calc
            viz = False # Visualize autoscaling process?
            test_img, test_dict = pm.auto_fullprocess(ref_image.point_dict, trans_image.point_dict, trans_image.image,verbose=True,visualize=viz)
            composed_img = compose(ref_image.image,test_img,30)
            if viz:
                cv.imshow("Map Comparison", composed_img)
                cv.waitKey()
            ref_dict, test_dict, coverage = pm.calc_coverage(ref_image.point_dict, test_dict)
            err_px, std_px, err_img = pm.calc_error(ref_dict, test_dict, composed_img)
            scores = dict()
            scores['name'] = image_list[image_selection].im_file[:-4]
            scores['scale'] = scale
            scores['error'] = err_px
            scores['std'] = std_px
            scores['coverage'] = coverage
            score_img = draw_resultsoverlay(err_img,scores)
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
