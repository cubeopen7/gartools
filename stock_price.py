# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from analyse.basic import missing
from dbwrapper.mongo import MongoClass

batch_size = 30
time_step = 20
hidden_size = 20
input_size = 7
output_size = 1
learning_rate = 0.01
max_epoch_size = 1000


def lstm(x, y=None, is_training=True):
    input_w = tf.get_variable("input_weight", shape=[input_size, hidden_size], dtype=tf.float32)
    input_b = tf.get_variable("input_bias", shape=[hidden_size], dtype=tf.float32)
    input = tf.reshape(x, shape=[-1, 7], name="unfold_input")
    input = tf.matmul(input, input_w) + input_b
    input = tf.reshape(input, shape=[batch_size, time_step, hidden_size], name="re_build")
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    init_states = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    output, final_states = tf.nn.dynamic_rnn(lstm_cell, input, initial_state=init_states, dtype=tf.float32)
    output = tf.reshape(output, shape=[-1, hidden_size], name="unfold_output")
    output_w = tf.get_variable("output_weight", shape=[hidden_size, output_size], dtype=tf.float32)
    output_b = tf.get_variable("output_bias", shape=[output_size], dtype=tf.float32)
    output = tf.matmul(output, output_w) + output_b
    loss = tf.reduce_mean(tf.square(tf.reshape(output, [-1]) - tf.reshape(y, [-1])))
    if not is_training:
        return None, loss
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_op, loss



if __name__ == "__main__":
    columns = ["code", "date", "open", "high", "low", "close", "amount", "volume", "turnover"]
    field_dict = {name: 1 for name in columns}
    field_dict["_id"] = 0

    # market_coll = MongoClass().market_coll
    # train_data = pd.DataFrame(list(market_coll.find({"code": "000001"}, field_dict))).sort_values(by="date", ascending=True)
    # train_data.to_csv("./data/stock_raw.csv", index=False)

    raw_data = pd.read_csv("./data/stock_raw.csv")
    num_data = raw_data[["open", "high", "low", "close", "amount", "volume", "turnover"]]
    num_data["label"] = 0
    num_data["label"][:num_data.shape[0]-1] = num_data["close"][1:]
    num_data = num_data.drop(num_data.shape[0]-1)

    def rolling_normalize(data):
        return ((data - data.mean()) / data.std())[-1]

    for col in num_data:
        col_data = num_data[col]
        num_data[col] = col_data.rolling(60).apply(rolling_normalize)
    num_data = num_data.dropna(axis=0)

    train_data = num_data[:5000]
    test_data = num_data[5000:]
    train_x = train_data.iloc[:, :7].values
    train_y = train_data.label.values
    test_x = test_data.iloc[:, :7].values
    test_y = test_data.label.values

    epoch_size = train_data.shape[0] // (batch_size * time_step)

    with tf.Graph().as_default():
        initializer = tf.random_normal_initializer(-0.1, 0.1)

        with tf.name_scope("train"):
            with tf.variable_scope("train_input"):
                x_input = tf.placeholder(dtype=tf.float32, name="train_x")
                y_input = tf.placeholder(dtype=tf.float32, name="train_y")
                i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
                x = tf.slice(x_input, [i * batch_size * time_step, 0], [batch_size * time_step, 7])
                x = tf.reshape(x, [batch_size, time_step, 7])
                y = tf.slice(y_input, [i * batch_size * time_step], [batch_size * time_step])
                y = tf.reshape(y, [batch_size, time_step])
            with tf.variable_scope("model", reuse=False, initializer=initializer):
                train_op, loss = lstm(x, y)

        with tf.name_scope("test"):
            with tf.variable_scope("test_input"):
                test_input_x = tf.placeholder(dtype=tf.float32, name="test_input_x")
                test_input_y = tf.placeholder(dtype=tf.float32, name="test_input_y")
                j = tf.train.range_input_producer(test_data.shape[0] // (batch_size * time_step), num_epochs=1, shuffle=False).dequeue()
                t_x = tf.slice(test_input_x, [j * batch_size * time_step, 0], [batch_size * time_step, 7])
                t_x = tf.reshape(t_x, [batch_size, time_step, 7])
                t_y = tf.slice(test_input_y, [j * batch_size * time_step], [batch_size * time_step])
                t_y = tf.reshape(t_y, [batch_size, time_step])
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                _, test_loss = lstm(t_x, t_y, is_training=False)

        train_feed_dict = {x_input: train_x, y_input: train_y}
        test_feed_dict = {test_input_x: test_x, test_input_y: test_y}

        sv = tf.train.Supervisor(logdir="./model/stock.model")
        with sv.managed_session() as sess:
            for i in range(max_epoch_size * epoch_size):
                _, cost = sess.run([train_op, loss], feed_dict=train_feed_dict)
                print("Epoch {}: Loss is {}.".format(i + 1, cost))

            cost = 0
            for i in range(test_data.shape[0] // (batch_size * time_step)):
                _cost = sess.run([test_loss], feed_dict=test_feed_dict)
                cost += _cost[0]
                print("Epoch {}: Loss is {}.".format(i + 1, cost / (i + 1)))

            sv.saver.save(sess, save_path="./model/stock.model/model", global_step=sv.global_step)