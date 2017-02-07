# coding: utf-8

import os
import json
import click
import numpy as np
import pandas as pd
from materials import iris_dataset
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = 'models'
MODEL_STRUCT_FILE = 'iris_struct.json'
MODEL_WEIGHTS_FILE = 'iris_weights.h5'

IRIS_DATA = iris_dataset()
IRIS_DATA = IRIS_DATA.reindex(np.random.permutation(IRIS_DATA.index))
IRIS_X = IRIS_DATA[IRIS_DATA.columns[:4]].as_matrix()
IRIS_Y = pd.get_dummies(IRIS_DATA[IRIS_DATA.columns[4]]).values


def build_model_from_file():
    model_struct_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_STRUCT_FILE)
    model_weights_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_WEIGHTS_FILE)

    model = model_from_json(open(model_struct_file, 'r').read())
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
    model.load_weights(model_weights_file)

    return model


def build_model():
    """建立一个 3 层(含输入层和输出层)神经网络"""
    model = Sequential()
    model.add(Dense(input_dim=4, output_dim=9, activation="relu"))
    model.add(Dense(output_dim=3, activation="softmax"))

    sgd = SGD(lr=0.1, decay=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return model


def save_model_to_file(model):
    # save model structure
    model_struct = model.to_json()
    model_struct_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_STRUCT_FILE)
    open(model_struct_file, 'w').write(model_struct)

    # save model weights
    model_weights_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_WEIGHTS_FILE)
    model.save_weights(model_weights_file, overwrite=True)


def train_xor(model):
    model.fit(IRIS_X, IRIS_Y, nb_epoch=300)

    return model


@click.command()
@click.argument('action')
def main(action):
    if action == 'build':
        click.echo("building model from scratch...")
        model = build_model()
        click.echo("train with iris data...")
        train_xor(model)
        click.echo("saving to hard disk...")
        save_model_to_file(model)
        click.echo("finished!")
    else:
        click.echo("invalid action.")


if __name__ == '__main__':
    main()
