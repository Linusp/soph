import os

import click

from demos.consts import PROJECT_ROOT, MODEL_PATH
from demos.sequence.pig_latin import (
    train as train_piglatin_model,
    test as test_piglatin_model,
)
from demos.sequence.adder import (
    train as train_adder_model,
    test as test_adder_model,
)


@click.group()
def main():
    pass


@main.command()
@click.option('--epoch', default=50, help='number of epoch to train model')
@click.option('-m', '--model-path', default=os.path.join(PROJECT_ROOT, MODEL_PATH),
              help='model files to save')
def train_piglatin(epoch, model_path):
    train_piglatin_model(epoch, model_path)


@main.command()
@click.option('-m', '--model-path', default=os.path.join(PROJECT_ROOT, MODEL_PATH),
              help='model files to read')
@click.argument('word')
def test_piglantin(model_path, word):
    test_piglatin_model(model_path, word)


@main.command()
@click.option('--epoch', default=50, help='number of epoch to train model')
@click.option('-m', '--model-path', default=os.path.join(PROJECT_ROOT, MODEL_PATH),
              help='model files to save')
def train_adder(epoch, model_path):
    train_adder_model(epoch, model_path)


@main.command()
@click.option('-m', '--model-path', default=os.path.join(PROJECT_ROOT, MODEL_PATH),
              help='model files to read')
@click.argument('expression')
def test_piglantin(model_path, expression):
    test_adder_model(model_path, expression)


if __name__ == '__main__':
    main()
