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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def normalization_0255(img):
    heatmap = img.copy()
    heatmap = heatmap - heatmap.min()
    print('heatmap.shape', heatmap.shape)
    heatmap /= heatmap.max()
    heatmap *= 255
    return heatmap

def build_mega_model(input_image_ph):
    prediction_1 = HourglassModel().fridaymodel(input_image_ph, is_training=False)
    return prediction_1
        
def hourglass_preprocessing(input_img):
    input_img = input_img[:, :, ::-1]
    input_img = np.array(input_img, dtype=np.float32)
    input_img /= 255
    input_img = np.expand_dims(input_img, 0)
    return input_img

def mapping_op(input):
    input = input.replace("module.", "") # since with tf.name_scope('module')
    l = input.split('.')
    s = np.size(l)
    prefix_l = l[:s-4]
    suffix_l = l[-4:]
    prefix    = '/'.join(prefix_l)
    suffix  = '.'.join(suffix_l)
    suffix_ = convert_suffix_torch2tf_para(suffix)
    out = prefix + '/' + suffix_
    return out

class ImageByDepth_Converter():
    def __init__(self, output_path=""):
        print("------ImageByDepth_Converter Init------")
        



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("OK")

    """Create the model and start the evaluation process."""
    global args
    args = get_arguments()
   

    input_video_path = ""

    Video_Converter = ImageByDepth_Converter()
    Video_Converter.run([input_video_path])
    print("pass")
    return 1


if __name__ == "__main__":
    main()
