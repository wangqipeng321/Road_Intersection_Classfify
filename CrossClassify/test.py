# -*- coding: UTF-8 -*-
"""
Filename: Network.py
Function:
    Test trained Network
Author: To_Fourier, CiWei
Created Time: 2020.07.14.13.34
Last Modified:
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from CrossClassify.Network import inference

NUM_CLASSES = 30
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 1

image_dir = 'C:\\Users\\44375\\Desktop\\CrossClassifyBIG\\Datasets_TEST'
log_dir = 'C:\\Users\\44375\\Desktop\\CrossClassifyBIG\\train_log'
cross_list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4',
              'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'F1', 'F2', 'F3',
              'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'I1', 'I2', 'I3', 'I4']
test_images_path = []


def get_file(file_dir):
    for i in range(100):
        test_images_path.append(file_dir + '\\' + str(i+1) + '.jpg')
    return test_images_path


def get_class(index):
    if index <= 2:
        index = 0
    elif index <= 5:
        index = 1
    elif index <= 9:
        index = 2
    elif max_index <= 13:
        index = 3
    elif index <= 16:
        index = 4
    elif index <= 19:
        index = 5
    elif index <= 22:
        index = 6
    elif index <= 25:
        index = 7
    elif index <= 29:
        index = 8
    return index


images = get_file(image_dir)
test_times = len(images)
test_right_count = 0
with tf.Graph().as_default():
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    h_fc3, logits = inference(x, 1, NUM_CLASSES)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading Success!')

    for j in range(test_times):
        image_test = Image.open(images[j])
        image_test = image_test.resize([IMAGE_WIDTH, IMAGE_HEIGHT], Image.ANTIALIAS)
        image_test = tf.image.per_image_standardization(image_test)
        with sess.as_default():
            image_test = image_test.eval()
        image_test = np.array(image_test, dtype=np.float32)
        image_test = image_test.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

        prediction = sess.run(logits, feed_dict={x: image_test})

        max_index = np.argmax(prediction)
        print('Predicted Cross: ', cross_list[max_index], 'with index', end=': ')
        max_index = get_class(max_index)
        print(max_index + 1)

    sess.close()
