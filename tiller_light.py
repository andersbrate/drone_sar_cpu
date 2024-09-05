
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from shutil import copyfile
import glob
import cv2
import time



"""
note; up is down, tiles start from bottom and move upwards left to right.
x1, y1 is bottom left, x2,y2 is top right
(0,0)
|------------------(x2,y2)|
|                         |
|                         |
|                         |
|                         |
|                         |
(x1,y1)-------------------|
                         (len(x), len(y)
"""
#NOTE! pillow image shape; widt, height = imr.shape




def tiller_light(image, overlap_percentage = 0, tiles = (3,2)):

    imr = np.array(image, dtype=np.uint8)

    img_width = imr.shape[1] #this is correct. stop doubting
    img_height = imr.shape[0]



    slice_size = int(img_width/tiles[0]), int(img_height/tiles[1])
    overlap = (int(overlap_percentage*slice_size[0]), int(overlap_percentage*slice_size[1]))

    num_slices = int(tiles[0]*tiles[1])
    outimages = []

    for i in range(tiles[0]):
        for j in range(tiles[1]):
            x1 = (i*slice_size[0])-overlap[0]
            y1 = img_height - (j*slice_size[1])+overlap[1]
            x2 = ((i+1)*slice_size[0])+overlap[0]
            y2 = (img_height - (j+1)*slice_size[1]-overlap[1])
            add_x = overlap[0] #amount of pixels which are added in width
            add_y = overlap[1] #amount of pixels which are added in height

            #handle edges which don't have overlap
            if j == 0:
                y1 = img_height
                add_y = 0 #If adding pixels on bottom of image it doesn't affect existing indexes
            elif j == tiles[1]-1:
                y2 = 0
                add_y = overlap[1]

            if i == 0:
                x1 = 0
                add_x = overlap[0]
            elif i == tiles[0]-1:
                x2 = img_width
                add_x = 0 #if adding pixels on right side it doesn't affect existing boxes


            sliced = imr[y2:y1, x1:x2]
            outimages.append((sliced, (x1,y2)))


    return outimages



if __name__ == '__main__':
    #params
    overlap = 15 #percentage
    tiles = (3,2)

    # get all image names
    #imnames = glob.glob('/home/POLITIET/abr063/git/drone_sar/tilling_test/test_dataset/images/*.jpg')
    imnames = ['/home/POLITIET/abr063/git/drone_sar/tilling_test/test_dataset/images/210529_Carnation_Enterprise_VIS_0025_00000378.jpg']


    image = cv2.imread(imnames[0])[...,::-1]

    print(imnames)

    tiller_light(image, overlap/100, tiles)


