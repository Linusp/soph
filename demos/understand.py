# coding: utf-8
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import GRU


def understand_return_sequence():
    """用来帮助理解 recurrent layer 中的 return_sequences 参数"""
    model_1 = Sequential()
    model_1.add(GRU(input_dim=256, output_dim=256, return_sequences=True))
    model_1.compile(loss='mean_squared_error', optimizer='sgd')
    train_x = np.random.randn(100, 78, 256)
    train_y = np.random.randn(100, 78, 256)
    model_1.fit(train_x, train_y, verbose=0)

    model_2 = Sequential()
    model_2.add(GRU(input_dim=256, output_dim=256, return_sequences=False))
    model_2.compile(loss='mean_squared_error', optimizer='sgd')
    train_x = np.random.randn(100, 78, 256)
    train_y = np.random.randn(100, 256)
    model_2.fit(train_x, train_y, verbose=0)

    inz = np.random.randn(100, 78, 256)
    rez_1 = model_1.predict_proba(inz, verbose=0)
    rez_2 = model_2.predict_proba(inz, verbose=0)

    print()
    print('=========== understand return_sequence =================')
    print('Input shape is: {}'.format(inz.shape))
    print('Output shape of model with `return_sequences=True`: {}'.format(rez_1.shape))
    print('Output shape of model with `return_sequences=False`: {}'.format(rez_2.shape))
    print('====================== end =============================')


def understand_variable_length_handle():
    """用来帮助理解如何用 recurrent layer 处理变长序列"""
    model = Sequential()
    model.add(GRU(input_dim=256, output_dim=256, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    train_x = np.random.randn(100, 78, 256)
    train_y = np.random.randn(100, 78, 256)
    model.fit(train_x, train_y, verbose=0)

    inz_1 = np.random.randn(1, 78, 256)
    rez_1 = model.predict_proba(inz_1, verbose=0)

    inz_2 = np.random.randn(1, 87, 256)
    rez_2 = model.predict_proba(inz_2, verbose=0)

    print
    print '=========== understand variable length ================='
    print 'With `return_sequence=True`'
    print 'Input shape is: {}, output shae is {}'.format(inz_1.shape, rez_1.shape)
    print 'Input shape is: {}, output shae is {}'.format(inz_2.shape, rez_2.shape)
    print '====================== end ============================='


def try_variable_length_train():
    """变长序列训练实验

    实验失败，这样得到的 train_x 和 train_y 的 dtype 是 object 类型，
    取其 shape 得到的是 (100,) ，这将导致训练出错
    """
    model = Sequential()
    model.add(GRU(input_dim=256, output_dim=256, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='sgd')

    train_x = []
    train_y = []
    for i in range(100):
        seq_length = np.random.randint(78, 87+1)
        sequence = []
        for _ in range(seq_length):
            sequence.append([np.random.randn() for _ in range(256)])

        train_x.append(np.array(sequence))
        train_y.append(np.array(sequence))

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    model.fit(np.array(train_x), np.array(train_y))


def try_variable_length_train_in_batch():
    """变长序列训练实验(2)"""
    model = Sequential()
    model.add(GRU(input_dim=256, output_dim=256, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # 分作两个 batch, 不同 batch 中的 sequence 长度不一样
    seq_lens = [78, 87]
    for i in range(2):
        train_x = np.random.randn(20, seq_lens[i], 256)
        train_y = np.random.randn(20, seq_lens[i], 256)
        model.train_on_batch(train_x, train_y)


if __name__ == '__main__':
    understand_return_sequence()
    understand_variable_length_handle()
