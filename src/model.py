import tensorflow as tf
import numpy as numpy
import cv2
import numpy as np
from re import sub as re_sub


tf.compat.v1.disable_eager_execution()


def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize(x, [h*ratio,w*ratio])

class Inception_Base(tf.keras.Model):
    def __init__(self,config, index):
        super(Inception_Base,self).__init__()
        filt = config[0]
        out_a = config[1]
        out_b = config[2]

        self.conv1 = tf.keras.layers.Conv2D(out_a,1,1, name='conv2d_{}'.format(index))
        self.bn1 = tf.keras.layers.BatchNormalization(center=False,scale=False, name='batch_normalization{}'.format(index))

        self.conv2 = tf.keras.layers.Conv2D(out_b,filt,1,padding='same', name='conv2d_{}'.format(index+1))
        self.bn2 = tf.keras.layers.BatchNormalization(center=False,scale=False, name='batch_normalization{}'.format(index+1))

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        return x




class Inception(tf.keras.Model):
    def __init__(self,config):
        super(Inception,self).__init__()
        stride = 1
        out_layers = config[0][0]
        kernel_size = 1
        
        self.conv0 = tf.keras.layers.Conv2D(out_layers,1,1, name='conv2d')
        self.bn0 = tf.keras.layers.BatchNormalization(center=False, scale=False, name='batch_normalization')

        self.inception_1 = Inception_Base(config[1],1)
        self.inception_2 = Inception_Base(config[2],3)
        self.inception_3 = Inception_Base(config[3],5)

    def call(self, x):
        out0 = self.conv0(x)
        out0 = self.bn0(out0)
        out0 = tf.nn.relu(out0)

        out1 = self.inception_1(x)
        out2 = self.inception_2(x)
        out3 = self.inception_3(x)

        out = tf.concat([out0,out1,out2,out3],3)

        return out

class Channel1(tf.keras.Model):
    def __init__(self):
        super(Channel1,self).__init__()
        self.inception0_0 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception0_1 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])

        
        
        self.pool = tf.keras.layers.AveragePooling2D(2, 2)
        
        self.inception1_0 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        
        self.inception1_1 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        
        self.inception1_2 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])

    def call(self,x):
        with tf.name_scope('0/0'):
            with tf.name_scope('0'):
                out0 = self.inception0_0(x)
            with tf.name_scope('1'):
                out0 = self.inception0_1(out0)
                
        with tf.name_scope('0/1'):
            with tf.name_scope('0'):
                pool = self.pool(x)
            with tf.name_scope('1'):
                out1 = self.inception1_0(pool)
            with tf.name_scope('2'):
                out1 = self.inception1_1(out1)
            with tf.name_scope('3'):
                out1 = self.inception1_2(out1)
            with tf.name_scope('4'):
                out1 = upsample_nn(out1,2)

        out = out0 + out1

        return out

class Channel2(tf.keras.Model):
    def __init__(self):
        super(Channel2,self).__init__()

        self.inception0_0 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception0_1 = Inception([[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])

        self.pool = tf.keras.layers.AveragePooling2D(2, 2)

        self.inception1_0 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception1_1 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.channel = Channel1()
        self.inception1_2 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception1_3 = Inception([[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])

    def call(self, x):
        with tf.name_scope('0/0'):
            with tf.name_scope('0'):
                out0 = self.inception0_0(x)
            with tf.name_scope('1'):
                out0 = self.inception0_1(out0)

        with tf.name_scope('0/1'):
            with tf.name_scope('0'):
                pool = self.pool(x)
            with tf.name_scope('1'):
                out1 = self.inception1_0(pool)
            with tf.name_scope('2'):
                out1 = self.inception1_1(out1)
            with tf.name_scope('3'):
                out1 = self.channel(out1)
            with tf.name_scope('4'):
                out1 = self.inception1_2(out1)
            with tf.name_scope('5'):
                out1 = self.inception1_3(out1)
            with tf.name_scope('6'):
                out1 = upsample_nn(out1, 2)

        out = out1 + out0
        return out


class Channel3(tf.keras.Model):
    def __init__(self):
        super(Channel3,self).__init__()

        self.pool = tf.keras.layers.AveragePooling2D(2, 2)

        self.inception0_0 = Inception([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception0_1 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.channel = Channel2()
        self.inception0_2 = Inception([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception0_3 = Inception([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])

        self.inception1_0 = Inception([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception1_1 = Inception([[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])

    def call(self, x):
        with tf.name_scope('0/0'):
            with tf.name_scope('0'):
                pool = self.pool(x)
            with tf.name_scope('1'):
                out1 = self.inception0_0(pool)
            with tf.name_scope('2'):
                out1 = self.inception0_1(out1)
            with tf.name_scope('3'):
                out1 = self.channel(out1)
            with tf.name_scope('4'):
                out1 = self.inception0_2(out1)
            with tf.name_scope('5'):
                out1 = self.inception0_3(out1)
            with tf.name_scope('6'):
                out1 = upsample_nn(out1, 2)
        
        with tf.name_scope('0/1'):
            with tf.name_scope('0'):
                out0 = self.inception1_0(x)
            with tf.name_scope('1'):
                out0 = self.inception1_1(out0)
        
        out = out1 + out0
        return out

class Channel4(tf.keras.Model):
    def __init__(self):
        super(Channel4,self).__init__()

        self.pool = tf.keras.layers.AveragePooling2D(2, 2)

        self.inception0_0 = Inception([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception0_1 = Inception([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.channel = Channel3()
        self.inception0_2 = Inception([[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]])
        self.inception0_3 = Inception([[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]])

        self.inception1_0 = Inception([[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
        #self.inception1_1 = Inception([[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])

    def call(self, x):
        with tf.name_scope('0/0'):
            with tf.name_scope('0'):
                pool = self.pool(x)
            with tf.name_scope('1'):
                out1 = self.inception0_0(pool)
            with tf.name_scope('2'):
                out1 = self.inception0_1(out1)
            with tf.name_scope('3'):
                out1 = self.channel(out1)
            with tf.name_scope('4'):
                out1 = self.inception0_2(out1)
            with tf.name_scope('5'):
                out1 = self.inception0_3(out1)
            with tf.name_scope('6'):
                out1 = upsample_nn(out1, 2)

        with tf.name_scope('0/1'):
            with tf.name_scope('0'):
                out0 = self.inception1_0(x)
        #out0 = self.inception1_1(out0)
        out = out1 + out0
        return out

class Hourglass(tf.keras.Model):
    def __init__(self, normalize=False):
        super(Hourglass,self).__init__()
        out_layers_1a = 128
        kernel_size_1a = 7
        stride_1a = 1
        out_layers_3a = 1
        kernel_size_3a = 3
        stride_3a = 1

        self.normalize = normalize

        self.conv0 = tf.keras.layers.Conv2D(out_layers_1a,kernel_size_1a,stride_1a,padding='same', name='conv2d')
        self.bn0 = tf.keras.layers.BatchNormalization()

        self.channel = Channel4()

        self.conv1 = tf.keras.layers.Conv2D(out_layers_3a,kernel_size_3a,stride_3a,padding='same', name='conv2d')

    def call(self,x, preTrainedWeights=None):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Keyword Arguments:
            preTrainedWeights {[type]} -- This is the python dictionary used by loadValues function (default: {None})
 
        Returns:
            [type] -- [description]
        """
        # pre-processing
        if self.normalize == True:
            x = tf.compat.v1.scalar_mul(1/255,x)
        # fridaymodel
        with tf.name_scope('module'):
            with tf.name_scope('0'):
               out = self.conv0(x)
            with tf.name_scope('1'):
                out = self.bn0(out)
            with tf.name_scope('2'):
               out = tf.nn.relu(out)
            with tf.name_scope('3'):
                out = self.channel(out)
            with tf.name_scope('4'):
                out = self.conv1(out)

            #post-processing
            out = tf.squeeze(out) # (1, 384, 512, 1)
            out = tf.math.exp(out)
            pred_depth = tf.math.divide(1.0, out)
            pred_depth = tf.math.divide(pred_depth, tf.compat.v1.reduce_max(pred_depth))
            
            if(preTrainedWeights):
                self.loadValues(preTrainedWeights)

        return pred_depth

    def getVariablePath(self, opPath):
        """
            returns the path of the operation without the
            inception, channel and module path polution
        """
        s1 = str(opPath).replace('module/', '')
        s2 = re_sub('channel\d+/', '', s1)
        s3 = re_sub('inception_*(base_)*\d*/*', '', s2)
        s4 = s3.replace('base/', '')
        s5 = s4.replace(':0','')
        return s5
    
    def loadValues(self, preTrainedWeights):
        """ This is called after the `call` method in order for
            the graph to be ready.

        Arguments:
            preTrainedWeights {dictionary with values numpy arrays} -- This is a python dictionary where the keys 
                                                                      are the paths in the checkpoint file and the values are numpy arrays 
                                                                      holding the pre trained weights for the trainable values of this module 
        """
        for trainableVar in tf.compat.v1.trainable_variables('module'):
            ckpt_path = self.getVariablePath(trainableVar.name)
            trainableVar.value = (preTrainedWeights[ckpt_path])


# IMG_FILE_TYPE = "png"
# MEGA_MODEL_WEIGHTS = './megadepth_model_fuse_bn_name/mega_prepost.ckpt'
# import os
# if __name__ == "__main__":
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     (train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
#     with tf.device("/cpu:0"):
#         trainedWeights = np.load('weights.pkl.npy',allow_pickle=True)[()]
        
#         H = Hourglass()
#         output = H.call(x, trainedWeights)
#         with tf.compat.v1.Session().as_default():
#             output.eval()
        # H.fit(...)
        # H.predict(...)
#         x = tf.zeros([4,14,14,3], tf.float32)
#         C1 = Channel1()
#         C1.call(x)
#         sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto())
#         init = tf.compat.v1.global_variables_initializer()
#         model_restore_var = [v for v in tf.compat.v1.global_variables()]
#         print(model_restore_var)
#         loader = tf.compat.v1.train.Saver(var_list=model_restore_var)
#         load(loader, sess, MEGA_MODEL_WEIGHTS)