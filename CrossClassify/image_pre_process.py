# -*- coding: UTF-8 -*-
"""
Filename: image_pre_process.py
Function:
    prepare training data and batch
Author: To_Fourier
Created Time: 2020.07.14.10.03
Last Modified: 2020.07.21.14.42
"""
import os
import math
import numpy as np
import tensorflow as tf


def get_file(file_dir, train_data_proportion=0.8):
    """
    :param file_dir: data sets path
    :param train_data_proportion: training data proportion
    :return train_images
    :return train_labels
    :return val_images
    :return val_labels
    """
    A1 = []  # cross 0
    label_A1 = []
    A2 = []
    label_A2 = []
    A3 = []
    label_A3 = []

    B1 = []  # cross 1
    label_B1 = []
    B2 = []
    label_B2 = []
    B3 = []
    label_B3 = []

    C1 = []  # cross 2
    label_C1 = []
    C2 = []
    label_C2 = []
    C3 = []
    label_C3 = []
    C4 = []
    label_C4 = []


    D1 = []  # cross 3
    label_D1 = []
    D2 = []
    label_D2 = []
    D3 = []
    label_D3 = []
    D4 = []
    label_D4 = []

    E1 = []  # cross 4
    label_E1 = []
    E2 = []
    label_E2 = []
    E3 = []
    label_E3 = []

    F1 = []  # cross 5
    label_F1 = []
    F2 = []
    label_F2 = []
    F3 = []
    label_F3 = []

    G1 = []  # cross 6
    label_G1 = []
    G2 = []
    label_G2 = []
    G3 = []
    label_G3 = []

    H1 = []  # cross 7
    label_H1 = []
    H2 = []
    label_H2 = []
    H3 = []
    label_H3 = []

    I1 = []  # cross 8
    label_I1 = []
    I2 = []
    label_I2 = []
    I3 = []
    label_I3 = []
    I4 = []
    label_I4 = []

    # cross 0
    for file in os.listdir(file_dir + '\\A1'):
        A1.append(file_dir + '\\A1\\' + file)
        label_A1.append(0)
    for file in os.listdir(file_dir + '\\A2'):
        A2.append(file_dir + '\\A2\\' + file)
        label_A2.append(1)
    for file in os.listdir(file_dir + '\\A3'):
        A3.append(file_dir + '\\A3\\' + file)
        label_A3.append(2)
    # Cross 1
    for file in os.listdir(file_dir + '\\B1'):
        B1.append(file_dir + '\\B1\\' + file)
        label_B1.append(3)
    for file in os.listdir(file_dir + '\\B2'):
        B2.append(file_dir + '\\B2\\' + file)
        label_B2.append(4)
    for file in os.listdir(file_dir + '\\B3'):
        B3.append(file_dir + '\\B3\\' + file)
        label_B3.append(5)
    # Cross 2
    for file in os.listdir(file_dir + '\\C1'):
        C1.append(file_dir + '\\C1\\' + file)
        label_C1.append(6)
    for file in os.listdir(file_dir + '\\C2'):
        C2.append(file_dir + '\\C2\\' + file)
        label_C2.append(7)
    for file in os.listdir(file_dir + '\\C3'):
        C3.append(file_dir + '\\C3\\' + file)
        label_C3.append(8)
    for file in os.listdir(file_dir + '\\C4'):
        C4.append(file_dir + '\\C4\\' + file)
        label_C4.append(9)
    # Cross 3
    for file in os.listdir(file_dir + '\\D1'):
        D1.append(file_dir + '\\D1\\' + file)
        label_D1.append(10)
    for file in os.listdir(file_dir + '\\D2'):
        D2.append(file_dir + '\\D2\\' + file)
        label_D2.append(11)
    for file in os.listdir(file_dir + '\\D3'):
        D3.append(file_dir + '\\D3\\' + file)
        label_D3.append(12)
    for file in os.listdir(file_dir + '\\D4'):
        D4.append(file_dir + '\\D4\\' + file)
        label_D4.append(13)
    # Cross 4
    for file in os.listdir(file_dir + '\\E1'):
        E1.append(file_dir + '\\E1\\' + file)
        label_E1.append(14)
    for file in os.listdir(file_dir + '\\E2'):
        E2.append(file_dir + '\\E2\\' + file)
        label_E2.append(15)
    for file in os.listdir(file_dir + '\\E3'):
        E3.append(file_dir + '\\E3\\' + file)
        label_E3.append(16)
    # Cross 5
    for file in os.listdir(file_dir + '\\F1'):
        F1.append(file_dir + '\\F1\\' + file)
        label_F1.append(17)
    for file in os.listdir(file_dir + '\\F2'):
        F2.append(file_dir + '\\F2\\' + file)
        label_F2.append(18)
    for file in os.listdir(file_dir + '\\F3'):
        F3.append(file_dir + '\\F3\\' + file)
        label_F3.append(19)
    # Cross 6
    for file in os.listdir(file_dir + '\\G1'):
        G1.append(file_dir + '\\G1\\' + file)
        label_G1.append(20)
    for file in os.listdir(file_dir + '\\G2'):
        G2.append(file_dir + '\\G2\\' + file)
        label_G2.append(21)
    for file in os.listdir(file_dir + '\\G3'):
        G3.append(file_dir + '\\G3\\' + file)
        label_G3.append(22)
    # Cross 7
    for file in os.listdir(file_dir + '\\H1'):
        H1.append(file_dir + '\\H1\\' + file)
        label_H1.append(23)
    for file in os.listdir(file_dir + '\\H2'):
        H2.append(file_dir + '\\H2\\' + file)
        label_H2.append(24)
    for file in os.listdir(file_dir + '\\H3'):
        H3.append(file_dir + '\\H3\\' + file)
        label_H3.append(25)
    # Cross 8
    for file in os.listdir(file_dir + '\\I1'):
        I1.append(file_dir + '\\I1\\' + file)
        label_I1.append(26)
    for file in os.listdir(file_dir + '\\I2'):
        I2.append(file_dir + '\\I2\\' + file)
        label_I2.append(27)
    for file in os.listdir(file_dir + '\\I3'):
        I3.append(file_dir + '\\I3\\' + file)
        label_I3.append(28)
    for file in os.listdir(file_dir + '\\I4'):
        I4.append(file_dir + '\\I4\\' + file)
        label_I4.append(29)

    image_list = np.hstack((A1, A2, A3, B1, B2, B3, C1, C2, C3, C4, D1, D2, D3, D4, E1, E2, E3,
                            F1, F2, F3, G1, G2, G3, H1, H2, H3, I1, I2, I3, I4))
    label_list = np.hstack((label_A1, label_A2, label_A3, label_B1, label_B2, label_B3,
                            label_C1, label_C2, label_C3, label_C4, label_D1, label_D2, label_D3, label_D4,
                            label_E1, label_E2, label_E3, label_F1, label_F2, label_F3,
                            label_G1, label_G2, label_G3, label_H1, label_H2, label_H3,
                            label_I1, label_I2, label_I3, label_I4))

    image_label = np.array([image_list, label_list])
    image_label = image_label.transpose()
    np.random.shuffle(image_label)

    all_image_list = list(image_label[:, 0])
    all_label_list = list(image_label[:, 1])

    num_all_data = len(all_label_list)
    num_train = int(math.ceil(num_all_data * train_data_proportion))

    train_images = all_image_list[0:num_train]
    train_labels = all_label_list[0:num_train]
    train_labels = [int(float(i)) for i in train_labels]

    val_images = all_image_list[num_train:num_all_data]
    val_labels = all_label_list[num_train:num_all_data]
    val_labels = [int(float(i)) for i in val_labels]

    return train_images, train_labels, val_images, val_labels


def get_batch(image_list, label_list, image_width, image_height, batch_size, capacity):
    """
    :param image_list: image path and file
    :param label_list: corresponding image label
    :param image_width: target width
    :param image_height: target height
    :param batch_size: batch size
    :param capacity: queue size
    :return image_batch: image batch to train
    :return label_batch: label batch to train
    """
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_content = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.resize_images(image, [image_width, image_height], method=0)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity, min_after_dequeue=1)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
