# %%
# from IPython import get_ipython


# %%
import tensorflow as tf
import numpy as np
import cv2
import sys, os
from skimage import io
from skimage.transform import resize
sys.path.append('..')

from src.model import Hourglass as Hourglass
import tools.inspect_checkpoint
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')


# %%
img_path = '../doc/demo.jpg'
out_path = '../doc/output_colored.png'


# %%
img = np.float32(io.imread(img_path))
w, h = img.shape[:2]
input_height = 240
input_width = 320


# %%
img = cv2.resize(img, (input_width, input_height))/255.0
img = np.expand_dims(img, axis=0)



# %%
sample_out = io.imread('../doc/hell0_demo_tf_320x240_prepost.png')
print(sample_out)


# %%
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print("Gpus are ",gpus)


# %%
x = tf.zeros([1,240,320,3], tf.float32)
weight_path = '../megadepth_model_fuse_bn_name/numpy_weights/weights_prepost.npy'
trainedWeights = np.load(weight_path, allow_pickle=True)[()]


# %%
if gpus:
    print("weight loaded")
    H = Hourglass(training=False, weightsPath=weight_path, normalize=False)
    print("Model created")
    H.trainable = False
    output = H.predict_on_batch(img)
    #print("output Finished",output)
    output = output*255
    # print(output)
    io.imsave(out_path, output)


# %%
K = trainedWeights.keys()
# print(trainedWeights['0/conv2d/kernel'])
# for k in K:
#     if "conv2d" in k:
#         print(k)
# print(K)


# %%
#trainedWeights2 = np.load('../weights.pkl.npy', allow_pickle=True)[()]


# %%
#K = trainedWeights2.keys()
#print(K)

