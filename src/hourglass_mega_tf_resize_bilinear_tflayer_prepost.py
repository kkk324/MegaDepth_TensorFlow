import tensorflow as tf
import os
import cv2
import numpy as np


class HourglassModel(object):

    def __init__(self):
        print('init work oh !')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''

        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.compat.v1.variable_scope(op_name, reuse=False):

                # Python3: dict has no module iteritems

                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name, shape=data.shape)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        #return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])
        return tf.image.resize(x, [h * ratio, w * ratio])


    def inception(self, x, config, is_training=False):
        stride = 1
        out_layers = config[0][0]
        kernel_size = 1

        conv0 = tf.compat.v1.layers.conv2d(x, out_layers, kernel_size, stride)
        conv0 = tf.compat.v1.layers.batch_normalization(conv0, center=False, scale=False)
        conv2_0 = tf.nn.relu(conv0)

        convs_list = []
        convs_list.append(conv2_0)

        for i in range(1, len(config)):
            # with tf.compat.v1.variable_scope('inception_' + str(i)):
            filt = config[i][0]
            out_a = config[i][1]  # no. of layer 1
            out_b = config[i][2]  # no. of layer 2
            # print ("out_a")
            # print (out_a)
            # print ("out_b")
            # print (out_b)
            conv1 = tf.compat.v1.layers.conv2d(x, out_a, 1, stride)  # kernel size = 1
            conv1 = tf.compat.v1.layers.batch_normalization(conv1, center=False, scale=False)
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.compat.v1.layers.conv2d(conv1, out_b, filt, stride, padding='same')  # kernel size = filt
            conv2 = tf.compat.v1.layers.batch_normalization(conv2, center=False, scale=False)
            conv2 = tf.nn.relu(conv2)
            convs_list.append(conv2)
            # print (tf.shape(conv2))

        convs_all = tf.concat([convs_list[0], convs_list[1], convs_list[2], convs_list[3]], 3)
        return convs_all


    def Channels1(self, x, is_training=False):
        with tf.compat.v1.variable_scope('0/0'):
            with tf.compat.v1.variable_scope('0'):
                conv1a = self.inception(x, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('1'):
                conv2a = self.inception(conv1a, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)

        with tf.compat.v1.variable_scope('0/1'):
            with tf.compat.v1.variable_scope('0'):
                pool1b = tf.compat.v1.layers.average_pooling2d(x, 2, 2)
            with tf.compat.v1.variable_scope('1'):
                conv2b = self.inception(pool1b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('2'):
                conv3b = self.inception(conv2b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('3'):
                conv4b = self.inception(conv3b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('4'):
                uconv4b = self.upsample_nn(conv4b, 2)

        output = conv2a + uconv4b
        return output


    def Channels2(self, x, is_training=False):
       with tf.compat.v1.variable_scope('0/0'):
            with tf.compat.v1.variable_scope('0'):
                conv1a = self.inception(x, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('1'):
                conv2a = self.inception(conv1a, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]], is_training=is_training)

       with tf.compat.v1.variable_scope('0/1'):
                with tf.compat.v1.variable_scope('0'):
                    pool1b = tf.compat.v1.layers.average_pooling2d(x, 2, 2)
                with tf.compat.v1.variable_scope('1'):
                    conv2b = self.inception(pool1b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
                with tf.compat.v1.variable_scope('2'):
                    conv3b = self.inception(conv2b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
                with tf.compat.v1.variable_scope('3'):
                    conv4b = self.Channels1(conv3b, is_training=is_training)
                with tf.compat.v1.variable_scope('4'):
                    conv5b = self.inception(conv4b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
                with tf.compat.v1.variable_scope('5'):
                    conv6b = self.inception(conv5b, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]], is_training=is_training)
                with tf.compat.v1.variable_scope('6'):
                    uconv6b = self.upsample_nn(conv6b, 2)

       output = conv2a + uconv6b
       return output

    def Channels3(self, x, is_training=False):
        with tf.compat.v1.variable_scope('0/0'):
            with tf.compat.v1.variable_scope('0'):
                pool1b = tf.compat.v1.layers.max_pooling2d(x, 2, 2)
            with tf.compat.v1.variable_scope('1'):
                conv2b = self.inception(pool1b, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('2'):
                conv3b = self.inception(conv2b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('3'):
                conv4b = self.Channels2(conv3b, is_training=is_training)
            with tf.compat.v1.variable_scope('4'):
                conv5b = self.inception(conv4b, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]], is_training=is_training)
            with tf.compat.v1.variable_scope('5'):
                conv6b = self.inception(conv5b, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('6'):
                uconv6b = self.upsample_nn(conv6b, 2)

        with tf.compat.v1.variable_scope('0/1'):
            with tf.compat.v1.variable_scope('0'):
                conv1a = self.inception(x, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('1'):
                conv2a = self.inception(conv1a, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]], is_training=is_training)
        output = conv2a + uconv6b
        return output

    def Channels4(self, x, is_training=False):
        with tf.compat.v1.variable_scope('0/0'):
            with tf.compat.v1.variable_scope('0'):
                pool1b = tf.compat.v1.layers.max_pooling2d(x, 2, 2)
            with tf.compat.v1.variable_scope('1'):
                conv2b = self.inception(pool1b, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('2'):
                conv3b = self.inception(conv2b, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('3'):
                conv4b = self.Channels3(conv3b, is_training=is_training)
            with tf.compat.v1.variable_scope('4'):
                conv5b = self.inception(conv4b, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]], is_training=is_training)
            with tf.compat.v1.variable_scope('5'):
                conv6b = self.inception(conv5b, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]], is_training=is_training)
            with tf.compat.v1.variable_scope('6'):
                uconv6b = self.upsample_nn(conv6b, 2)

        with tf.compat.v1.variable_scope('0/1'):
            with tf.compat.v1.variable_scope('0'):
                conv1a = self.inception(x, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]], is_training=is_training)

        #with tf.compat.v1.variable_scope('OutCh4'):
            #output = tf.add(conv1a, uconv6b)
        output = conv1a + uconv6b
        return output



    def fridaymodel(self, input, is_training):
        out_layers_1a = 128
        kernel_size_1a = 7
        stride_1a = 1
        out_layers_3a = 1
        kernel_size_3a = 3
        stride_3a = 1

        # pre-processing
        input = tf.scalar_mul(1 / 255, input)
        # input = tf.image.resize_images(input, (384, 512))

        with tf.name_scope('module'):
            with tf.compat.v1.variable_scope('0'):
                conv1a = tf.compat.v1.layers.conv2d(input, out_layers_1a, kernel_size_1a, padding='same', activation=None,
                                          name='conv2d')

            with tf.compat.v1.variable_scope('1'):
                conv1a_bn = tf.compat.v1.layers.batch_normalization(conv1a)

            with tf.compat.v1.variable_scope('2'):
                conv1a_relu = tf.nn.relu(conv1a_bn)

            with tf.compat.v1.variable_scope('3'):
                conv2a = self.Channels4(conv1a_relu, is_training=is_training)

            with tf.compat.v1.variable_scope('4'):
                conv3a = tf.compat.v1.layers.conv2d(conv2a, out_layers_3a, kernel_size_3a, stride_3a, padding='same',
                                          activation=None, name='conv2d')
            # post-processing
            conv3a = tf.compat.v1.squeeze(conv3a)  # (1, 384, 512, 1)
            conv3a = tf.compat.v1.exp(conv3a)
            pred_inv_depth = tf.compat.v1.div(1.0, conv3a)
            pred_inv_depth = tf.compat.v1.div(pred_inv_depth, tf.reduce_max(pred_inv_depth))

        return  pred_inv_depth #conv3a

if __name__ == "__main__":
    pass
