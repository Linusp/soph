# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import GRU


def understand_return_sequence():
    """
    用来帮助理解 recurrent layer 中的 return_sequences 参数
    """
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

    print
    print '=========== understand return_sequence ================='
    print 'Input shape is: {}'.format(inz.shape)
    print 'Output shape of model with `return_sequences=True`: {}'.format(rez_1.shape)
    print 'Output shape of model with `return_sequences=False`: {}'.format(rez_2.shape)
    print '====================== end ============================='


def understand_variable_length_handle():
    """
    用来帮助理解如何用 recurrent layer 处理变长序列
    """
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


if __name__ == '__main__':
    understand_return_sequence()
    understand_variable_length_handle()
