# coding: utf-8
from __future__ import print_function

import os
import json
import click
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = 'models'
MODEL_STRUCT_FILE = 'xor_struct.json'
MODEL_WEIGHTS_FILE = 'xor_weights.h5'

TRAIN_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
TRAIN_Y = np.array([[0], [1], [1], [0]])


def build_model_from_file():
    model_struct_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_STRUCT_FILE)
    model_weights_file = os.path.join(PROJECT_ROOT, MODEL_PATH, MODEL_WEIGHTS_FILE)

    sgd = SGD(lr=0.1)

    model = model_from_json(open(model_struct_file, 'r').read())
    model.compile(loss="binary_crossentropy", optimizer=sgd)
    model.load_weights(model_weights_file)

    return model


def build_model():
    """建立一个 3 层(含输入层和输出层)神经网络"""
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=5, activation="tanh"))
    model.add(Dense(output_dim=1, activation="sigmoid"))

    sgd = SGD(lr=0.1)

    model.compile(loss="binary_crossentropy", optimizer=sgd)

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
    model.fit(TRAIN_X, TRAIN_Y, nb_epoch=500, batch_size=2)

    return model


@click.command()
@click.argument('action')
def main(action):
    if action == 'build':
        click.echo("building model from scratch...")
        model = build_model()
        click.echo("train with xor data...")
        train_xor(model)
        click.echo("saving to hard disk...")
        save_model_to_file(model)
        click.echo("finished!")
    elif action == 'test':
        model = build_model_from_file()
        while True:
            user_input = raw_input("Enter two number, 0 or 1, anything else to quit: ")
            user_input = [e.strip() for e in user_input.split()]
            if not all(e.isdigit() and int(e) in (0, 1) for e in user_input):
                break

            if len(user_input) != 2:
                break

            x = np.array([[int(e) for e in user_input[:2]]])
            print(model.predict(x))
    else:
        click.echo("invalid action.")


if __name__ == '__main__':
    main()
