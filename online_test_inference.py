# All Includes
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import sys

import os
from socket import *

##### inference 관련
win_size = 10
class_num = 3
data_length = 13
pdata_length = 34


def csv_read(csv_path, pdata_length):
    # csv_test 읽어오기
    res_test = []
    fr_test = open(csv_path, 'r', encoding='utf-8')
    rdr_test = csv.reader(fr_test)

    for line in rdr_test:
        # print(line)
        res_test.extend(line)
    fr_test.close()

    i_len2 = int(len(res_test) / pdata_length)
    j_len2 = int(len(res_test) / (pdata_length * win_size))

    test_arr = np.array(res_test).reshape(j_len2, win_size, pdata_length)  # 30, 34

    # 데이터 셋
    #     X_test = np.array(test_arr)
    #     print(X_test.shape)
    return np.array(test_arr)


# #################### Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
        #         dtype=np.str
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_


def pred_txt(pred, label_path):
    for i in range(0, len(pred)):
        with open(label_path, 'a') as test_txt:
            test_txt.write('%d\n' % (pred[i]))


def LSTM_RNN(_X, _weights, _biases):

    # LSTM Neural Network's internal structure

    n_hidden = 256  # Hidden layer num of features
    n_classes = class_num  # Total classes (should go up, or should go down)
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    #print(_X.shape)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    #print(_X.shape)
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)
    #print(_X.shape)
    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    #     outputs, states = tf.contrib.rnn.static_rnn(lstm_cell_1, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


if __name__ == '__main__':

    csv_path = ''

    if len(sys.argv) is 1:
        print('옵션을 주지 않고 스크립트 실행')

    for argv_num in range(len(sys.argv)):
        print('sys.argv[%d] = %s' % (argv_num, sys.argv[argv_num]))

    csv_path = sys.argv[1]

    # socket test
    #clientSock = socket(AF_INET, SOCK_STREAM)
    #clientSock.connect(('192.168.0.69', 5000))

    # Input Data
    # training_data_count = len(X_test)  # 7352 training series (with 50% overlap between each serie)
    # test_data_count = len(X_test)  # 2947 testing series
    # n_steps = len(X_test[0])  # 128 timesteps per series
    n_steps = 10  # 128 timesteps per series
    # n_input = len(X_test[0][0])  # 9 input parameters per timestep
    n_input = 34  # 9 input parameters per timestep
    n_classes = 3
    n_hidden = 256

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="x")
    y = tf.placeholder(tf.float32, [None, n_classes], name="y")

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name="weights_hidden"),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name="weights_out")
    }

    #print(type(weights['out']))

    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="bias_out")
    }

    pred = LSTM_RNN(x, weights, biases)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ###### SDUFall Dataset 영상 fall detection URFD + SDUFall

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # y_test_ = one_hot(y_test)
    ###################### ckpt_10f_34_1_3c_acc9880_URFD_SDUFall
    save_file = './ckpt_10f_34_1_3c_acc9880_URFD_SDUFall/train_model.ckpt'
    # save_file = './ckpt_33f_8_3c_acc9864_URFD_SDUF/train_model.ckpt'
    # save_file = './ckpt_30f_13_3c_acc9935/train_model.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_file)

        c_path = csv_path

        one_hot_predictions = sess.run(
            pred,
            feed_dict={
                x: csv_read(c_path, 34),
                #             y: y_test_
            }
        )

        predictions = one_hot_predictions.argmax(1)
        print('%d frame, class : %d, feature_num : %d' % (win_size, class_num, pdata_length))
        print('%s 추론 결과' % (c_path))
        print(predictions)

        print(' ')
        with open('/home/jeong/fall_detection_online/test/out/output.txt', 'a') as out_txt:
            csv_writer = csv.writer(out_txt)
            csv_writer.writerow(predictions)
        """
        win = 61
        #chk_alarm = 0

        for j in range(0, len(predictions)):
            start_index = j

            if (j + win) >= len(predictions):
                end_index = len(predictions)

            else:
                end_index = j + win

            #     print(end_index)
            #     print(predictions[j:end_index])
            if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
                if j != (end_index - 1):
                    #chk_alarm = chk_alarm + 1
                    print('Fall Detection!!! %d index' % (j + 1))
                    # socket test
                    #clientSock.send('warning from online_test_inference.py'.encode('utf-8'))
        """
