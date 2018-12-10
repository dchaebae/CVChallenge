rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os

from tensorflow.contrib import layers
from skimage.io import imread, imsave

DATASET_PATH = '/home/agoyal/cos429challenge'
BATCH_SIZE = 8
SUMMARIES_PATH = '/home/agoyal/cos429challenge/summaries'
MAX_ITERATION = 1000

DUMP_FOLDER = '/home/agoyal/cos429challenge/prediction'

if not osp.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)


def build_model():
    color = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    mask = tf.placeholder(dtype=tf.bool, shape=[None, 128, 128])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

    conv1 = layers.conv2d(color, 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)
    conv2 = layers.conv2d(conv1, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)
    conv3 = layers.conv2d(conv2, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)
    predict = layers.conv2d(conv3, 3, [1, 1], 1, 'same', activation_fn=None)

    predict_n = tf.nn.l2_normalize(predict, dim=3)
    target_n = tf.nn.l2_normalize(target, dim=3)

    cosine_angle = tf.reduce_sum(predict_n * target_n, axis=3)
    loss = -tf.reduce_mean(tf.boolean_mask(cosine_angle, mask))

    return color, mask, target, predict_n, loss


def load_train_data(iteration, batch_size):
    total = 20000
    start = (iteration * batch_size) % total

    color_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([batch_size, 128, 128], dtype=np.uint8)
    target_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)

    for i in range(batch_size):
        color_path = osp.join(DATASET_PATH, 'train', 'color', '{}.png'.format(i + start))
        mask_path = osp.join(DATASET_PATH, 'train', 'mask', '{}.png'.format(i + start))
        target_path = osp.join(DATASET_PATH, 'train', 'normal', '{}.png'.format(i + start))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        target_npy[i, ...] = imread(target_path)

    target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy, target_npy


def train():
    color, mask, target, predict_n, loss = build_model()
    loss_summ = tf.summary.scalar('training_loss', loss)
    writer = tf.summary.FileWriter(SUMMARIES_PATH)

    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    with tf.device("/gpu:0"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(MAX_ITERATION):
                color_npy, mask_npy, target_npy = load_train_data(i, BATCH_SIZE)
                feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy
                }
                loss_val, summ, _ = sess.run([loss, loss_summ, train_op], feed_dict=feed_dict)
                writer.add_summary(summ, i)

                print(loss_val)

            for i in range(20000):
                color_npy, mask_npy, target_npy = load_train_data(i, 1)
                feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy
                }
                predict_val, = sess.run([predict_n], feed_dict={color: color_npy, target: target_npy})

                predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                imsave(osp.join(DUMP_FOLDER, '{}.png'.format(i)), predict_img)


if __name__ == '__main__':
    train()