from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf


BATCH_SIZE = 128

class MyFCN:
    def __init__(self):
        pass

    def _activation_summary(x):
      tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
      tf.histogram_summary(tensor_name + '/activations', x)
      tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    def _variable_with_weight_decay(name, shape, stddev, wd):
        """
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        """       
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable(name, shape, initializer=initializer)
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def inference(images):
        """Build the fcn model.
        Args:
            images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
            labels: may be added in future.
        Returns:
            Logits.
        """
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[11, 11, 3, 96],
                                         stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='VALID')
            biases = tf.get_variable('biases',[96], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
            _activation_summary(conv1)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
            padding='VALID', name='pool1')
        #-------------------------------------------
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',shape=[5,5,96,256],
                stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, strides=[1,2,2,1], padding='SAME')
            biases = tf.get_variable('biases',[256],initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        #-------------------------------------------
        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay('weights',shape=[3,3,256,384],
                stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(pool2, kernel, strides=[1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases',[384],initializer=tf.constant_initializer(0,1))
            conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        #-------------------------------------------
        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3,3,384,384],
                stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(conv3, kernel, strides=[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',[384],initializer=tf.constant_initializer(0.1))
            conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        #-------------------------------------------
        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3,3,384,256],
                stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(conv4, kernel, strides=[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',[256],initializer=tf.constant_initializer(0.1))
            conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        # pool5
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        #-------------------------------------------
        #fc6
        with tf.variable_scope('fc6') as scope:
            reshape = tf.reshape(pool5, [BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights',
                shape=[dim,4096],stddev=0.04,wd=0.004)
            biases = tf.get_variable('biases',[4096],
                initializer=tf.constant_initializer(0.1))
            fc6 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
            _activation_summary(fc6)
        # drop layer may be need
        #----------------------------------------------
        # fc7
        with tf.variable_scope('fc7') as scope:
            weights = _variable_with_weight_decay('weights', 
                shape=[4096, 4096], stddev=0.04, wd=0.004)
            biases = tf.get_variable('biases',[4096],initializer=tf.constant_initializer(0.1))
            fc7 = tf.nn.relu(tf.matmul(fc6,weights)+biases, name=scope.name)
            _activation_summary(fc7)
        
        # upsample image
        with tf.variable_scope('upsample') as scope:
            im_size = image.get_shape().as_list()
            conv7 = tf.reshape(fc7,[BATCH_SIZE,1,1,4096],'conv7')
            kernel = _variable_with_weight_decay('weights', shape=[im_size[0],im_size[1],4,4096],
                stddev=1e-4, wd=0.0)
            output_shape = tf.pack([BATCH_SIZE,im_size[0],im_size[1],4])
            deconv = tf.nn.conv2d_transpose(conv7, kernel, output_shape, strides=[1,1,1,1],padding='SAME')
def loss(prediction_image, depth_image):
    """Add L2Loss to all the trainable variables.
    Args:
        logits: fcn_image from inference.  [batch_size, w , h , c]
        depth_image: the depth label from grey_image, which size is same as logits
                     [batch_size, w, h, c]
    Returns:
        Loss tensor of type float
    """
    predict = tf.reshape(prediction_image, [-1])
    labels = tf.reshape(depth_image, [-1])
    cost = tf.nn.l2_loss(predict - labels, name='l2loss')/BATCH_SIZE
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_less')
 
