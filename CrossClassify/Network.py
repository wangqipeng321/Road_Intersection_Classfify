# -*- coding: UTF-8 -*-
"""
Filename: Network.py
Function:
    Neural Network Design:
        3 convolution layers with pooling
        2 fully-connected layers
        activate function: ReLU
        softmax for classification
Author: To_Fourier, CiWei
Created Time: 2020.07.13.11.17
Last Modified: 2020.07.21.14.35
"""
import tensorflow as tf


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial


def conv2d(image, filter_size):
    return tf.nn.conv2d(image, filter_size, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(image, name):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def inference(images, batch_size, num_classes):
    """
    :param images:
    :param batch_size:
    :param num_classes: number of image classes
    :return h_fc3: output off last fully connected layer
    :return softmax_linear: output of softmax_linear
    """
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 32], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1) + b_conv1, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
        norm1 = max_pool_2x2(h_conv1, 'pooling1')

    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 32, 64], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2, name=scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = max_pool_2x2(h_conv2, 'pooling2')

    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 64, 256], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([256]), name='biases')
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3) + b_conv3, name=scope.name)

    with tf.variable_scope('pooling3_lrn') as scope:
        norm3 = max_pool_2x2(h_conv3, 'pooling3')

    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 4096], 0.005), name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([4096]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

    with tf.variable_scope('fc2') as scope:
        w_fc2 = tf.Variable(weight_variable([4096, 256], 0.005), name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([256]), name='biases', dtype=tf.float32)
        h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        h_fc2 = tf.nn.dropout(h_fc2, keep_prob=0.5)

    with tf.variable_scope('fc3') as scope:
        w_fc3 = tf.Variable(weight_variable([256, num_classes], 0.005), name='weights', dtype=tf.float32)
        b_fc3 = tf.Variable(bias_variable([num_classes]), name='biases', dtype=tf.float32)
        h_fc3 = tf.matmul(h_fc2, w_fc3) + b_fc3

    with tf.variable_scope('softmax_linear') as scope:
        softmax_linear = tf.nn.softmax(h_fc3, name=scope.name)

    return h_fc3, softmax_linear


def losses(logits, labels):
    """
    :param logits: network output value
    :param labels: real value, 0 or 1
    :return loss
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss


def training(loss, learning_rate):
    """
    :param loss: loss value from function losses
    :param learning_rate: learning rate
    :return train_op: train operation
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    :param logits: network output value
    :param labels: real value, 0 or 1
    :return accuracy
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
