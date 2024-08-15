import os
import datetime
import cv2
from math import radians, sin, cos

# I/O FUNCTIONS ==============================
def generate_unique_filename(label="capture",format="%y-%m-%d-%H-%M-%S",extension=".png"):
    # Get the current time
    now = datetime.datetime.now()
    # Format the time string 
    time_str = now.strftime(format)
    # Create the filename with the desired suffix
    filename = f"{time_str}-{label}{extension}"
    return filename

def save_screenshot(image, save_dir='./output'):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Generate the file name and path
    im_name = generate_unique_filename()
    save_path = os.path.join(save_dir,im_name)
    # Save the image to the specified file path
    cv2.imwrite(save_path, image)
    print(f"Screenshot saved: {save_path}")


# POINT I/O FUNCTIONS ==========================================
def get_map_label(pts_file):
    '''
    Loads a csv textfile containing a maps reference points and
    returns the first line of the file as the "label" for the map
    '''
    print("Getting label for "+str(pts_file)+" ...")
    label = "None"
    try:
        f = open(pts_file,"r")
        lines = f.readlines()
        label = lines[0].split(',')[0]
    
        print(f"Map Label: {label}")
    except:
        print("GET LABEL FAILED")
    
    return label

def import_pts(pts_file):
    print("Importing "+str(pts_file)+" ...")
    try:
        point_dict = dict()
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
            x = float(data[1])
            y = float(data[2][:-1])
            coords = [x,y]
            point_dict[marker] = coords
        print("Imported point dict:")
        print(point_dict)
        return point_dict
    except:
        print("PTS IMPORT FAILED")
        return None
    
def export(dir,filename,ref_dict):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        pts_file = filename+".pts"
        f = open(os.path.join(dir,pts_file),"w+")
        f.write(filename+"\n")
        for p in sorted(ref_dict.keys()):
            write_str = str(p)+","+str(ref_dict[p][0])+","+str(ref_dict[p][1])+"\n"
            f.write(write_str)
        f.close()
        print("POINT DICT EXPORTED TO " + filename)
    except:
        print("Unable to export points")


# DRAWING FUNCTIONS ============================================================================
def draw_textline(image,string,org=[5,30],scale=1.0,c=(0,0,0),c2=(255,255,255),t=2,border=True):
    font = cv2.FONT_HERSHEY_PLAIN
    if border: image = cv2.putText(image,string,org,font,scale,c2,t+1,cv2.LINE_AA)
    image = cv2.putText(image,string,org,font,scale,c,t,cv2.LINE_AA)

def draw_ref_pt(image,pos,label=None,color=(0,0,255)):
    # print("Drawing Point \"{}\" at {}".format(label,pos))
    pos[0] = int(pos[0])
    pos[1] = int(pos[1])
    # cv2.circle(im, center, 3, (0, 25, 255), 2)
    d = 10
    t = 1
    cv2.line(image,(pos[0],pos[1]-d),(pos[0],pos[1]+d),color,t)
    cv2.line(image,(pos[0]-d,pos[1]),(pos[0]+d,pos[1]),color,t)
    if label is not None:
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        lbl_offset = [2,-5]
        org = (pos[0]+lbl_offset[0],pos[1]+lbl_offset[1])
        image = cv2.putText(image,str(label),org,font,font_scale,(255,255,255),thickness+1,cv2.LINE_AA)
        image = cv2.putText(image,str(label),org,font,font_scale,color,thickness,cv2.LINE_AA)
    
    return image

def scale_image_with_aspect_ratio(image, scale_factor):
    if image is None:
        raise ValueError("Image not found or cannot be read.")

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions while preserving the aspect ratio
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height))
    # cv2.imshow("Orig",image)
    # cv2.imshow("Resize",resized_image)
    # cv2.waitKey(0)
    return resized_image

def compose(image1, image2, blend=50):
    alpha = float(blend) / 100
    beta = 1.0 - alpha

    blended = cv2.addWeighted(image1, alpha, image2, beta, 0.0)