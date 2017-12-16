# coding: utf-8
from __future__ import print_function

import os
import click
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = 'models'

TRAIN_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
TRAIN_Y = np.array([[0], [1], [1], [0]])


def build_model_from_file(model_file):
    structure, weights = pickle.load(open(model_file, 'rb'))
    model = Sequential.from_config(structure)
    model.set_weights(weights)

    return model


def build_model():
    """建立一个 3 层(含输入层和输出层)神经网络"""
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=5, activation="tanh"))
    model.add(Dense(output_dim=1, activation="sigmoid"))

    sgd = SGD(lr=0.1)

    model.compile(loss="binary_crossentropy", optimizer=sgd)

    return model


def save_model_to_file(model, model_file):
    # save model structure
    structure = model.get_config()
    weights = model.get_weights()
    pickle.dump((structure, weights), open(model_file, 'wb'))


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
        save_model_to_file(model, os.path.join(PROJECT_ROOT, MODEL_PATH, 'xor.model'))
        click.echo("finished!")
    elif action == 'test':
        model = build_model_from_file(os.path.join(PROJECT_ROOT, MODEL_PATH, 'xor.model'))
        while True:
            user_input = input("Enter two number, 0 or 1, anything else to quit: ")
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
