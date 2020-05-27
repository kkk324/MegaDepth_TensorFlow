from __future__ import print_function


import time
import glob
# from tqdm import tqdm
import argparse
import threading
import pprint
import pickle
import os
import sys

from skimage import io
from skimage.transform import resize
import cv2
from hourglass_mega_tf_resize_bilinear_tflayer_prepost import HourglassModel


import numpy as np
import tensorflow as tf
import cv2

args = None

IMG_FILE_TYPE = "png"
MEGA_MODEL_WEIGHTS = './megadepth_model_fuse_bn_name/mega_prepost.ckpt'


# Generate video from images
def generate_video():
    image_folder = './depth'
    video_name   = 'video_from_depths_KITTI.avi'

    images = [img for img in os.listdir(image_folder)
                 if img.endswith(".jpg") or
                    img.endswith(".jpeg") or
                    img.endswith(".png")]
    
    images.sort() 
    path = os.path.join(image_folder,images[0])
    print(path)
    frame = cv2.imread(path)
   
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #fucking important!
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    # For imshow image 
    #img2 = cv2.imread(os.path.join(image_folder, images[0]))
    #cv2.imshow('My Image', img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    for image in images:
        #print(os.path.join(image_folder, image))
        video.write(cv2.imread(os.path.join(image_folder, image)))	
        #print(image)
    
    cv2.destroyAllWindows()
    video.release()




def main():
    generate_video()	


if __name__ == "__main__":
    main()
