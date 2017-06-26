# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

'''
APIs Operating TFRecords.
'''

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def data_features(data_list):
    feature = {}
    for info in data_list:
        name = info.get("name")
        dtype = info.get("type")
        data =info.get("data")
        _f_data = None
        if dtype == "str":
            _f_data = bytes_feature(data)
        elif dtype == "int":
            _f_data = int64_feature(data)
        elif dtype == "float":
            _f_data = float_feature(data)
        feature[name] = _f_data
    return tf.train.Features(feature=feature)

def data_example(data_list):
    return tf.train.Example(features=data_features(data_list))

def read_int64_feature():
    return tf.FixedLenFeature([], tf.int64)

def read_bytes_feature():
    return tf.FixedLenFeature([], tf.string)

def read_float_feature():
    return tf.FixedLenFeature([], tf.float32)

def get_param_features(feature_info):
    features = {}
    for info in feature_info:
        fixed_feature = None
        if info[1] == "str":
            fixed_feature = read_bytes_feature()
        elif info[1] == "int":
            fixed_feature = read_int64_feature()
        elif info[1] == "float":
            fixed_feature = read_float_feature()
        features[info[0]] = fixed_feature
    return features

def get_features(serialized_example, features):
    features = get_param_features(features)
    features = tf.parse_single_example(serialized_example, features=features)
    return features



'''
APIs Transfer Images to TFRecord.
'''

def img2tfrecord(img_dir, img_width, img_height, tfrecord_name, img_label=None):
    if os.path.exists(tfrecord_name):
        raise FileExistsError("The TFRecord file already exists.")
    tic = time.time()
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    if img_label is None:
        for img_name in tqdm(os.listdir(img_dir)):
            img_path = img_dir + img_name
            img = Image.open(img_path)
            img_data = np.array(img.resize((img_width, img_height), Image.ANTIALIAS))  # Image.ANTIALIAS的作用是抗锯齿
            data_list = [{"name": "img_raw", "type": "str", "data": img_data.tostring()}]
            example = data_example(data_list)
            writer.write(example.SerializeToString())
    else:
        for img_name in tqdm(os.listdir(img_dir)):
            img_path = img_dir + img_name
            idx = int(img_name.split(".")[0])
            label = img_label[img_label["name"]==idx]["label"].values[0]
            img = Image.open(img_path)
            img_data = np.array(img.resize((img_width, img_height), Image.ANTIALIAS))  # Image.ANTIALIAS的作用是抗锯齿
            data_list = [{"name": "label", "type": "int", "data": label},
                         {"name": "img_raw", "type": "str", "data": img_data.tostring()}]
            example = data_example(data_list)
            writer.write(example.SerializeToString())
    writer.close()
    print("Data transfer process done in {}s".format(time.time()-tic))

def read_tfrecord_op(file_list, img_width, img_height, num_epochs=1, with_label=False, normalize=False, **kwargs):
    file_queue = tf.train.string_input_producer(file_list, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    if with_label:
        feature = [("img_raw", "str"), ("label", "int")]
    features = get_features(serialized_example, features=feature)
    image = tf.decode_raw(features["img_raw"], tf.int8)
    image = tf.reshape(image, shape=[img_width, img_height, 3])
    if normalize:
        image = (tf.cast(image, tf.float32) / 255.0 - 0.5) * 2
    label = tf.cast(features["label"], tf.int32)
    images, labels = tf.train.shuffle_batch([image, label], **kwargs)
    return images, labels
