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

tf.compat.v1.disable_eager_execution()

args = None

IMG_FILE_TYPE = "png"
MEGA_MODEL_WEIGHTS = '../megadepth_model_fuse_bn_name/mega_prepost.ckpt'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="one view video to depth v03")
    parser.add_argument("--img_path", type=str, default="./input2.png",
                        help="Path to the image.")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="Select GPU ID.")
    return parser.parse_args()

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

# def mapping_op(input):
#     input = input.replace("module.", "") # since with tf.name_scope('module')
#     l = input.split('.')
#     s = np.size(l)
#     prefix_l = l[:s-4]
#     suffix_l = l[-4:]
#     prefix    = '/'.join(prefix_l)
#     suffix  = '.'.join(suffix_l)
#     suffix_ = convert_suffix_torch2tf_para(suffix)
#     out = prefix + '/' + suffix_
#     return out

def main():
    """Create the model and start the evaluation process."""
    global args
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # hourglass_imag_pl = tf.placeholder(tf.float32, (1, 240, 320, 3))

    # input_height = 384
    # input_width = 512
    input_height = 240
    input_width = 320

    imag_pl = tf.compat.v1.placeholder(tf.float32, (1, input_height, input_width, 3))
    #imag_pl = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
    mega_out = build_mega_model(imag_pl)


    

    # Set up TF session and initialize variables. 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Load weights.
    model_restore_var = [v for v in tf.compat.v1.global_variables()]
    loader = tf.compat.v1.train.Saver(var_list=model_restore_var)
    load(loader, sess, MEGA_MODEL_WEIGHTS)

  
    ###############################################################################
    print('-------------------Single input------------------------------')
    img_path = '../doc/demo.jpg'
    print(img_path)

    ############## pre-processing ###############
    # img = np.float32(io.imread(img_path)) / 255.0
    ############## pre-processing ###############

    # turn on when on pre-processing
    img = np.float32(io.imread(img_path))

    w, h = img.shape[:2]

    # turn on when on pre-processing
    # img = resize(img, (input_height, input_width), order=1)

    img = cv2.resize(img, (input_width, input_height))
    img = np.expand_dims(img, axis=0)

    j=0
    for k in range(10):
        start_time = time.time()
        depth = sess.run(mega_out,
                         feed_dict={imag_pl: img})
        duration = time.time() - start_time
        print('duration_' + str(j) + ':', duration)
        j=j+1
    depth = np.squeeze(depth)  # (1, 384, 512, 1)
    pred_inv_depth = depth
    ###############################################################################

    ############################### post ##########################
    # depth = np.exp(depth)
    #
    # # np.save(file='demo_tf', arr=depth)
    #
    # pred_inv_depth = 1 / depth
    # pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)
    # pred_inv_depth = cv2.resize(pred_inv_depth, (h, w))
    # ############################################################

    # pred_inv_depth = cv2.resize(depth, (h, w))

    print('----------------- output path --------------------')
    output_path = 'hell0_demo_tf_320x240_prepost.png'
    #pred_inv_depth = depth
    io.imsave(output_path, pred_inv_depth)

    print('Output is:', output_path)

    # for i in range(16):
    #     input_buff_list.append(input_image)
    # input_batch_ori = np.stack(input_buff_list)
    #
    # start_time = time.time()
    # depth = sess.run(fusion_out,
    #     feed_dict={imag_pl:input_batch_ori})
    # duration1 = time.time() - start_time
    # print('ckpt_duration1', duration1)
    #
    #
    # # time profile
    # for i in range(11):
    #   i=i+1
    #   start_time = time.time()
    #   depth = sess.run(fusion_out,
    #                    feed_dict={imag_pl: input_batch_ori})
    #   duration2 = time.time() - start_time
    #   print('ckpt_duration_per_frame_' + str(i)+':', duration2/16)
    #
    # # depth = depth[0, :, :, 0]
    # # depth = normalization_0255(depth.max() - depth)
    # # out_img = hist.HistogramMapping(depth)
    # # out_img = cv2.resize(out_img, (h, w))
    # for i in range(16):
    #     cv2.imwrite('fusion_out_img_'+ str(i) +'.png', depth[i])

    print('Done Here!!')


if __name__ == "__main__":
    main()
