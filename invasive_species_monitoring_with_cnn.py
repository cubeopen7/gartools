# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.nets import vgg

from tools.tf.data import img2tfrecord, read_tfrecord_op

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "D:/Code/data/Invasive Species Monitoring/", "Datas' base folder direction.")
flags.DEFINE_string("train_dir", "train/", "Train data folder's name.")
flags.DEFINE_string("test_dir", "test/", "Test data folder's name.")
flags.DEFINE_string("train_label_file", "train_labels.csv", "Train datas' label file.")
flags.DEFINE_string("train_tfrecord_name", "train.tfrecords", "Train data TFRecord file name.")
flags.DEFINE_string("test_tfrecord_name", "test.tfrecords", "Test data TFRecord file name.")
flags.DEFINE_string("model_file", "model.ckpt", "Store the parameters of the model.")
flags.DEFINE_integer("width", 224, "Images' width.")
flags.DEFINE_integer("height", 224, "Images' height.")
flags.DEFINE_integer("num_epochs", 10, "Repeat times of the total samples.")
flags.DEFINE_integer("batch_size", 32, "Training batch size.")
flags.DEFINE_integer("decay_per", 50, "Learning rate decay epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_float("decay_rate", 0.01, "Learning rate decay rate.")

_PER_BATCH = int(math.floor(2295 / FLAGS.batch_size))

# train_label = pd.read_csv(FLAGS.data_dir + FLAGS.train_label_file)
# train_label.columns = ["name", "label"]
# img2tfrecord(img_dir=FLAGS.data_dir+FLAGS.train_dir, img_width=FLAGS.width, img_height=FLAGS.height, tfrecord_name=FLAGS.train_tfrecord_name, img_label=train_label)  # 将train数据(图片和标签)转为TFRecord, 方便使用
# img2tfrecord(img_dir=FLAGS.data_dir+FLAGS.test_dir, img_width=FLAGS.width, img_height=FLAGS.height, tfrecord_name=FLAGS.test_tfrecord_name)  # 将test数据(图片和标签)转为TFRecord, 方便使用

def _create_cnn_graph(img, is_training=True):
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(img, 2, slim.conv2d, 64, [3, 3], scope="conv1")  # 创建多个拥有相同变量的指定层
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
            net = slim.max_pool2d(net, [2, 2], scope="pool2")
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope="conv3")
            net = slim.max_pool2d(net, [2, 2], scope="pool3")
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope="conv4")
            net = slim.max_pool2d(net, [2, 2], scope="pool4")
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope="conv5")
            net = slim.max_pool2d(net, [2, 2], scope="pool5")
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512, biases_regularizer=slim.l2_regularizer(0.0005), scope="fc1")
    net = slim.fully_connected(net, 2, activation_fn=None, biases_regularizer=slim.l2_regularizer(0.0005), scope="fc2")
    return net

def _losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def _optimize(loss):
    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=_PER_BATCH*FLAGS.decay_per, decay_rate=FLAGS.decay_rate, staircase=True)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op


# Training model
tf.reset_default_graph()
with tf.device("/cpu:0"):
    train_img, label = read_tfrecord_op([FLAGS.train_tfrecord_name], img_width=FLAGS.width, img_height=FLAGS.height, num_epochs=FLAGS.num_epochs, normalize=True, with_label=True,
                                        batch_size=FLAGS.batch_size, capacity=256, num_threads=2, min_after_dequeue=32)
    pred = _create_cnn_graph(train_img, is_training=True)
    loss = _losses(pred, label)
    train_op = _optimize(loss)

print("Start training...")
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(FLAGS.num_epochs):
        avg_loss, acc = 0, 0
        for j in range(_PER_BATCH):
            _, l = sess.run([train_op, loss])
            avg_loss += l / _PER_BATCH
            print("Epoch{} Batch {} - average loss: {}".format(i+1, j+1, avg_loss))

    coord.request_stop()
    coord.join(threads)
    print("Training complete.")

    saver = tf.train.Saver(slim.get_model_variables())
    saver.save(sess, FLAGS.model_file)


# tf.reset_default_graph()

# Use model to get the result with test samples