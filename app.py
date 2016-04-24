# coding: utf-8

import click
from flask import Flask, request, render_template


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/xor/', methods=['GET', 'POST'])
def xor():
    return '404 not implemented'


@app.route('/iris/', methods=['GET', 'POST'])
def iris():
    return '404 not implemented'


@click.command()
@click.option('--port', default=8000, help='port to listen')
def main(port):
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
