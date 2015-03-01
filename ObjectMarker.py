#!/usr/bin/python
 
###############################################################################
# Name          : ObjectMarker.py
# Author        : Python implementation: sqshemet 
#                 Original ObjectMarker.cpp: http://www.cs.utah.edu/~turcsans/DUC_files/HaarTraining/
# Date          : 7/24/12
# Description   : Object marker utility to be used with OpenCV Haar training. 
#                 Tested on Ubuntu Linux 10.04 with OpenCV 2.1.0.
# Usage         : python ObjectMarker.py outputfile inputdirectory
###############################################################################
 
import cv2 as cv
import numpy as np
import os
import sys , math
from PIL import Image
 
IMG_SIZE = (300,300)
IMG_CHAN = 3
IMG_DEPTH = cv.IPL_DEPTH_8U
image = np.zeros((300,300,3), np.uint8)# = cv.CreateImage(IMG_SIZE, IMG_DEPTH, IMG_CHAN)
image2 = np.zeros((300,300,3), np.uint8)# = cv.CreateImage(IMG_SIZE, IMG_DEPTH, IMG_CHAN) 
roi_x0 = 0
roi_y0 = 0
roi_x1 = 0
roi_y1 = 0
num_of_rec = 0
leye_x = 0
leye_y = 0
reye_x = 0
reye_y = 0
pick_leye = False
pick_reye = False
start_draw = False
window_name = "<Space> to save and load next, <X> to skip, <ESC> to exit."



def Distance(p1,p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def on_mouse(event, x, y, flag, params):
        global start_draw
        global roi_x0
        global roi_y0
        global roi_x1
        global roi_y1
        global pick_leye
        global pick_reye         
        global leye_x
        global leye_y
        global reye_x
        global reye_y
        if (event == cv.cv.CV_EVENT_LBUTTONDOWN):
                if (not start_draw and (not pick_leye) and (not pick_reye)):
                        roi_x0 = x
                        roi_y0 = y
                        start_draw = True
                elif pick_leye :
                        leye_x = x
                        leye_y = y
                        image2 = cv.cv.CloneImage(image)
                        cv.cv.Rectangle(image2, (x-1, y-1), (x,y), 
                                cv.cv.CV_RGB(255,0,255),1)
                        pick_reye = True
                        pick_leye = False
                elif pick_reye :
                        reye_x = x
                        reye_y = y
                        pick_reye = False
                else:
                        roi_x1 = x
                        roi_y1 = y
                        start_draw = False
                        pick_leye = True
        elif (event == cv.cv.CV_EVENT_MOUSEMOVE and start_draw):
                #Redraw ROI selection
                image2 = cv.cv.CloneImage(image)
                cv.cv.Rectangle(image2, (roi_x0, roi_y0), (x,y), 
                        cv.cv.CV_RGB(255,0,255),1)
                cv.cv.ShowImage(window_name, image2)
def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
        # calculate offsets in original image
        offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
        offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
        # get the direction
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
        # calc rotation angle in radians
        rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
        # distance between them
        dist = Distance(eye_left, eye_right)
        # calculate the reference eye-width
        reference = dest_sz[0] - 2.0*offset_h
        #scale factor
        scale = float(dist)/float(reference)
        # rotate original around the left eye
        image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
        # crop the rotated image
        crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
        crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
        image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
        # resize it
        image = image.resize(dest_sz, Image.ANTIALIAS)
        return image
def main():
 
        global image
        global roi_x0
        global roi_y0
        global roi_x1
        global roi_y1
        iKey = 0
        if (len(sys.argv) != 2):
                sys.stderr.write("%s raw/data/directory\n" % sys.argv[0])
                return -1
 
        input_directory = sys.argv[1]#"D:\documents\Polytech_2014-2015\Semestre_8\Projet\Christelle\\"#sys.argv[2]
        output_file = "output.dat"#sys.argv[1]
 
        #Get a file listing of all files within the input directory
        try:
                files = os.listdir(input_directory)
        except OSError:
                sys.stderr.write("Failed to open dirctory %s\n" 
                        % input_directory)
                return -1
 
        files.sort()
 
        sys.stderr.write("ObjectMarker: Input Directory: %s Output File %s\n" 
                        % (input_directory, output_file))
 
        # init GUI
        cv.namedWindow(window_name, 1)
        cv.cv.SetMouseCallback(window_name, on_mouse, None)
 
        sys.stderr.write("Opening directory...")
        # init output of rectangles to the info file
        os.chdir(input_directory)
        sys.stderr.write("done.\n")
 
        str_prefix = input_directory
        num = 0
        try:
                output = open(output_file, 'w')
        except IOError:
                sys.stderr.write("Failed to open file %s.\n" % output_file)
                return -1
        
        for file in files:
                str_postfix = ""
                num_of_rec = 0
                img = str_prefix + file
                num = num + 1
                sys.stderr.write("Loading image %s...\n" % img)
                
                try: 
                        image = cv.cv.LoadImage(img, 1)
                except IOError: 
                        sys.stderr.write("Failed to load image %s.\n" % img)
                        break
                        return -1
 
                #  Work on current image
                cv.cv.ShowImage(window_name, image)
                # Need to figure out waitkey returns.
                # <ESC> = 43            exit program
                # <Space> = 48          add rectangle to current image
                # <x> = 136             skip image
                iKey = cv.cv.WaitKey(0) % 255
                # This is ugly, but is actually a simplification of the C++.
                
                if iKey == 27:
                        cv.cv.DestroyWindow(window_name)
                        return 0
                elif iKey == 32:
                        num_of_rec += 1
                        # Currently two draw directions possible:
                        # from top left to bottom right or vice versa
                        print(roi_x0, roi_y0, roi_x1, roi_y1)
                        myimage = Image.open(img)
                        CropFace(myimage, eye_left=(leye_x,leye_y), eye_right=(reye_x,reye_y), offset_pct=(0.2,0.2), dest_sz=(92,112)).save(str_prefix+ str(num) +"_20_20_200_200.jpg")
                        CropFace(myimage, eye_left=(leye_x,leye_y), eye_right=(reye_x,reye_y), offset_pct=(0.3,0.3), dest_sz=(92,112)).save(str_prefix+ str(num) +"_30_30_200_200.jpg")
                        CropFace(myimage, eye_left=(leye_x,leye_y), eye_right=(reye_x,reye_y), offset_pct=(0.2,0.2)).save(str_prefix+ str(num) +"_20_20_70_70.jpg")
                        if (roi_x0<roi_x1 and roi_y0<roi_y1):
                                str_postfix += " %d %d %d %d\n" % (roi_x0,
                                        roi_y0, (roi_x1-roi_x0), (roi_y1-roi_y0))
                                #CropFace(myimage, eye_left=(roi_x0+((roi_x1-roi_x0)/4),(roi_y1-roi_y0)/4), eye_right=(roi_x0+(3*(roi_x1-roi_x0)/4),roi_y0+((roi_y1-roi_y0)/4)), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(str_prefix+ str(num) +"_10_10_200_200.jpg")
                        
                        elif (roi_x0>roi_x1 and roi_y0>roi_y1):
                                str_postfix += " %d %d %d %d\n" % (roi_x1, 
                                        roi_y1, (roi_x0-roi_x1), (roi_y0-roi_y1))
                                #CropFace(myimage, eye_left=(roi_x1+((roi_x0-roi_x1)/4),(roi_y0-roi_y1)/4), eye_right=(roi_x1+(3*(roi_x0-roi_x1)/4),roi_y1+((roi_y0-roi_y1)/4)), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(str_prefix+ str(num) +"_10_10_200_200.jpg")
                
                        elif (roi_x0<roi_x1 and roi_y0>roi_y1):
                                str_postfix += " %d %d %d %d\n" % (roi_x0, 
                                        roi_y1, (roi_x1-roi_x0), (roi_y0-roi_y1))
                                #CropFace(myimage, eye_left=(roi_x0+((roi_x1-roi_x0)/4),(roi_y0-roi_y1)/4), eye_right=(roi_x0+(3*(roi_x1-roi_x0)/4),roi_y1+((roi_y0-roi_y1)/4)), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(str_prefix+ str(num) +"_10_10_200_200.jpg")
                
                        elif (roi_x0>roi_x1 and roi_y0<roi_y1):
                                str_postfix += " %d %d %d %d\n" % (roi_x1, 
                                        roi_y0, (roi_x0-roi_x1), (roi_y1-roi_y0))
                                #CropFace(myimage, eye_left=(roi_x1+((roi_x0-roi_x1)/4),(roi_y1-roi_y0)/4), eye_right=(roi_x0+(3*(roi_x1-roi_x0)/4),roi_y0+((roi_y1-roi_y0)/4)), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(str_prefix+ str(num) +"_10_10_200_200.jpg")
                
                        output.write(file + " " + str(num_of_rec) + str_postfix)
                elif iKey == 136:
                        sys.stderr.write("Skipped %s.\n" % img)
                else:
                        print("erreur",iKey)
        
        cv.cv.DestroyWindow(window_name)
        return 0
                
if __name__ == '__main__':
        main()
