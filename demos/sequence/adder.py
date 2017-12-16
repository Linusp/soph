# coding: utf-8
from __future__ import print_function

import os
import numpy as np
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.models import Sequential

from ..utils import build_model_from_file, save_model_to_file

BEGIN_SYMBOL = '^'
END_SYMBOL = '$'
CHARSET = set('0123456789+ ' + BEGIN_SYMBOL + END_SYMBOL)
CHAR_NUM = len(CHARSET)
MAX_LEN = 12
MAX_LEN = 12

CHAR_TO_INDICES = {c: i for i, c in enumerate(CHARSET)}
INDICES_TO_CHAR = {i: c for c, i in CHAR_TO_INDICES.items()}


def vectorize(seq, seq_len, vec_size):
    vec = np.zeros((seq_len, vec_size), dtype=int)
    for i, ch in enumerate(seq):
        vec[i, CHAR_TO_INDICES[ch]] = 1

    for i in range(len(seq), seq_len):
        vec[i, CHAR_TO_INDICES[END_SYMBOL]] = 1

    return vec


def build_data():
    """生成所有三位数(包含)一下的加法"""
    plain_x = []
    plain_y = []
    for i in range(0, 100):
        for j in range(0, 100):
            x = BEGIN_SYMBOL + '{}+{}'.format(i, j) + END_SYMBOL
            y = BEGIN_SYMBOL + '{}'.format(i + j) + END_SYMBOL

            plain_x.append(x)
            plain_y.append(y)

    data_size = len(plain_x)

    # convert to one-hot
    X = np.zeros((data_size, MAX_LEN, CHAR_NUM), dtype=int)
    Y = np.zeros((data_size, MAX_LEN, CHAR_NUM), dtype=int)

    for i, seq in enumerate(plain_x):
        X[i] = vectorize(seq, MAX_LEN, CHAR_NUM)

    for i, seq in enumerate(plain_y):
        Y[i] = vectorize(seq, MAX_LEN, CHAR_NUM)

    return X, Y


def build_model(input_size, seq_len, hidden_size):
    """建立一个 seq2seq 模型"""
    model = Sequential()
    model.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(GRU(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="softmax")))
    model.compile(loss="categorical_crossentropy", optimizer='adam')

    return model


def train(epoch, model_path):
    train_x, train_y = build_data()

    model = build_model(CHAR_NUM, MAX_LEN, 128)
    model.fit(train_x, train_y, nb_epoch=epoch)

    model_file = os.path.join(model_path, "adder.model")
    save_model_to_file(model, model_file)


def test(model_path, expression):
    model_file = os.path.join(model_path, 'adder.model')
    model = build_model_from_file(model_file)

    x = np.zeros((1, MAX_LEN, CHAR_NUM), dtype=int)
    expression = BEGIN_SYMBOL + expression.lower().strip() + END_SYMBOL
    x[0] = vectorize(expression, MAX_LEN, CHAR_NUM)

    pred = model.predict(x)[0]
    print(''.join([
        INDICES_TO_CHAR[i] for i in pred.argmax(axis=1)
        if INDICES_TO_CHAR[i] not in (BEGIN_SYMBOL, END_SYMBOL)
    ]))
