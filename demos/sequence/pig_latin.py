# coding: utf-8
from __future__ import print_function

import os
import re
import string
from itertools import dropwhile

import numpy as np
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, RepeatVector

from ..consts import DATA_PATH
from ..utils import build_model_from_file, save_model_to_file

WORDS_FILE = 'words.txt'
BEGIN_SYMBOL = '^'
END_SYMBOL = '$'
CHAR_SET = set(string.ascii_lowercase + BEGIN_SYMBOL + END_SYMBOL)
CHAR_NUM = len(CHAR_SET)
CHAR_TO_INDICES = {c: i for i, c in enumerate(CHAR_SET)}
INDICES_TO_CHAR = {i: c for c, i in CHAR_TO_INDICES.items()}
MAX_INPUT_LEN = 18
MAX_OUTPUT_LEN = 20

NON_ALPHA_PAT = re.compile('[^a-z]')


def is_vowel(char):
    return char in ('a', 'e', 'i', 'o', 'u')


def is_consonant(char):
    return not is_vowel(char)


def pig_latin(word):
    if is_vowel(word[0]):
        return word + 'yay'
    else:
        remain = ''.join(dropwhile(is_consonant, word))
        removed = word[:len(word) - len(remain)]
        return remain + removed + 'ay'


def vectorize(word, seq_len, vec_size):
    vec = np.zeros((seq_len, vec_size), dtype=int)
    for i, ch in enumerate(word):
        vec[i, CHAR_TO_INDICES[ch]] = 1

    for i in range(len(word), seq_len):
        vec[i, CHAR_TO_INDICES[END_SYMBOL]] = 1

    return vec


def build_data():
    words_file = os.path.join(DATA_PATH, WORDS_FILE)
    words = [
        w.lower().strip() for w in open(words_file, 'r').readlines()
        if w.strip() != '' and not NON_ALPHA_PAT.findall(w.lower().strip())
    ]

    plain_x = []
    plain_y = []
    for w in words:
        plain_x.append(BEGIN_SYMBOL + w)
        plain_y.append(BEGIN_SYMBOL + pig_latin(w))

    # train_x 和 train_y 必须是 3-D 的数据
    train_x = np.zeros((len(words), MAX_INPUT_LEN, CHAR_NUM), dtype=int)
    train_y = np.zeros((len(words), MAX_OUTPUT_LEN, CHAR_NUM), dtype=int)
    for i in range(len(words)):
        train_x[i] = vectorize(plain_x[i], MAX_INPUT_LEN, CHAR_NUM)
        train_y[i] = vectorize(plain_y[i], MAX_OUTPUT_LEN, CHAR_NUM)

    return train_x, train_y


def build_model(input_size, seq_len, hidden_size):
    """建立一个 sequence to sequence 模型"""
    model = Sequential()
    model.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(GRU(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')

    return model


def train(epoch, model_path):
    x, y = build_data()
    indices = int(len(x) / 10)
    test_x = x[:indices]
    test_y = y[:indices]
    train_x = x[indices:]
    train_y = y[indices:]

    model = build_model(CHAR_NUM, MAX_OUTPUT_LEN, 128)

    model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=128, nb_epoch=epoch)

    model_file = os.path.join(model_path, 'pig_latin.model')
    save_model_to_file(model, model_file)


def test(model_path, word):
    model_file = os.path.join(model_path, 'pig_latin.model')
    model = build_model_from_file(model_file)

    x = np.zeros((1, MAX_INPUT_LEN, CHAR_NUM), dtype=int)
    word = BEGIN_SYMBOL + word.lower().strip() + END_SYMBOL
    x[0] = vectorize(word, MAX_INPUT_LEN, CHAR_NUM)

    pred = model.predict(x)[0]
    print(''.join([
        INDICES_TO_CHAR[i] for i in pred.argmax(axis=1)
        if INDICES_TO_CHAR[i] not in (BEGIN_SYMBOL, END_SYMBOL)
    ]))
