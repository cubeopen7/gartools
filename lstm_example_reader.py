# -*- coding: utf-8 -*-

import os
import collections
import tensorflow as tf


__all__ = ["ptb_raw_data"]


_TRAIN_NAME = "ptb.train.txt"
_VALID_NAME = "ptb.valid.txt"
_TEST_NAME = "ptb.test.txt"


def _read_words(file_name):
    # 返回字符串列表
    with tf.gfile.GFile(file_name, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(file_name):
    # 返回单词与编号的字典,以供查询
    data = _read_words(file_name)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: ( -x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(file_name, word_to_id):
    data = _read_words(file_name)
    id_list = [word_to_id[word] for word in data if word in word_to_id]
    return id_list


def ptb_raw_data(data_path):
    train_path = os.path.join(data_path, _TRAIN_NAME)
    valid_path = os.path.join(data_path, _VALID_NAME)
    test_path = os.path.join(data_path, _TEST_NAME)

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)  # 一维, 长度等于单词数量
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        return x, y
