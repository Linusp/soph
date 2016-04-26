# coding: utf-8

import os
import numpy as np
from keras.layers.core import Activation
from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential, model_from_json


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = 'models'
MODEL_STRUCT_FILE = 'adder_struct.json'
MODEL_WEIGHTS_FILE = 'adder_weights.h5'


SEP = '|'
BLANK = ' '
CHARSET = set('0123456789+ ' + SEP)
CHAR_NUM = len(CHARSET)
INPUT_SEQUENCE_LEN = 6
OUTPUT_SEQUENCE_LEN = 6

CHAR_TO_INDICES = {c: i for i, c in enumerate(CHARSET)}
INDICES_TO_CHAR = {i: c for c, i in CHAR_TO_INDICES.iteritems()}


def build_data(data_size):
    plain_x = []
    plain_y = []
    for _ in range(data_size):
        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        x = '{0}+{1}{2}  '.format(a, b, SEP)
        y = '{:5d}{}'.format(a+b, SEP)

        plain_x.append(x)
        plain_y.append(y)

    # convert to one-hot
    X = np.zeros((data_size, INPUT_SEQUENCE_LEN, CHAR_NUM), dtype=int)
    Y = np.zeros((data_size, OUTPUT_SEQUENCE_LEN, CHAR_NUM), dtype=int)

    for i, seq in enumerate(plain_x):
        for j, char in enumerate(seq):
            X[i, j, CHAR_TO_INDICES[char]] = 1

    for i, seq in enumerate(plain_y):
        for j, char in enumerate(seq):
            Y[i, j, CHAR_TO_INDICES[char]] = 1

    return X, Y


def build_model_from_file():
    model_struct_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_STRUCT_FILE)
    model_weights_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_WEIGHTS_FILE)

    model = model_from_json(open(model_struct_file, 'r').read())
    model.compile(loss="mse", optimizer='adam')
    model.load_weights(model_weights_file)

    return model


def build_model():
    """建立一个 3 层(含输入层和输出层)神经网络"""
    model = Sequential()
    model.add(GRU(input_dim=CHAR_NUM, output_dim=CHAR_NUM, return_sequences=True))
    model.add(Activation('tanh'))
    model.compile(loss="mse", optimizer='adam')

    return model


def save_model_to_file(model):
    # save model structure
    model_struct = model.to_json()
    model_struct_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_STRUCT_FILE)
    open(model_struct_file, 'w').write(model_struct)

    # save model weights
    model_weights_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_WEIGHTS_FILE)
    model.save_weights(model_weights_file, overwrite=True)


def train_adder(model):
    X, Y = build_data(1000)
    model.fit(X, Y, nb_epoch=1000)

    return model


if __name__ == '__main__':
    train = False
    test = not train
    if train:
        model = build_model()
        model = train_adder(model)
        save_model_to_file(model)

    if test:
        model = build_model_from_file()
        test_x, test_y = build_data(10)

        preds = model.predict(test_x)
        for i in range(len(test_x)):
            # print test_x[i].argmax(axis=1), preds[i].argmax(axis=1), test_y[i].argmax(axis=1)
            seq_in = ''.join([INDICES_TO_CHAR[k] for k in test_x[i].argmax(axis=1)])
            seq_out = ''.join([INDICES_TO_CHAR[k] for k in preds[i].argmax(axis=1)])
            print seq_in, seq_out
