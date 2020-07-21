# -*- coding: UTF-8 -*-
"""
Filename: Network.py
Function:
    Train designed network
Author: To_Fourier, CiWei
Created Time: 2020.07.13.21.04
Last Modified: 2020.07.21.14.32
"""

import os
import numpy as np
import tensorflow as tf

from CrossClassify.image_pre_process import get_file, get_batch
from CrossClassify.Network import inference, losses, training, evaluation

# hyper parameters
NUM_CLASSES = 30
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 128
CAPACITY = 128
MAX_STEP = 5000
learning_rate = 0.00001

data_set_dir = 'C:\\Users\\44375\\Desktop\\CrossClassifyBIG\\Datasets_TRAIN'
log_train_dir = 'C:\\Users\\44375\\Desktop\\CrossClassifyBIG\\train_log'            # path to load or save model
image_train_list, label_train_list, image_val_list, label_val_list = get_file(data_set_dir, 1)
image_batch, label_batch = get_batch(image_list=image_train_list, label_list=label_train_list, image_width=IMG_WIDTH,
                                     image_height=IMG_HEIGHT, batch_size=BATCH_SIZE, capacity=CAPACITY)

h_fc3, train_logits = inference(image_batch, BATCH_SIZE, NUM_CLASSES)
train_loss = losses(h_fc3, label_batch)
train_op = training(train_loss, learning_rate)
train_acc = evaluation(train_logits, label_batch)

summary_op = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(log_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading Success!')
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

try:
    for step in np.arange(MAX_STEP + 1):
        if coordinator.should_stop():
            break
        image_train_list, label_train_list, image_val_list, label_val_list = get_file(data_set_dir, 1)
        _, training_loss, training_accuracy = sess.run([train_op, train_loss, train_acc])

        if step % 100 == 0:
            print('Step %d, train loss = %.10f, train accuracy = %.2f%%' % (step, training_loss,
                                                                            training_accuracy * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            checkpoint_path = os.path.join(log_train_dir, 'thing.ckpt')
            saver.save(sess, checkpoint_path)
except tf.errors.OutOfRangeError:
    print('Finished Training!')
finally:
    coordinator.request_stop()

coordinator.join(threads)
sess.close()
