# coding: utf-8

import os
import sys
import heapq
import numpy as np
from materials import tang_poetries
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN

POETRIES = tang_poetries().dropna().content.tolist()
# POET_BEGIN = '^'
# POET_END = '$'
POET_DEL = '|'


def decode_to_unicode(text):
    return text if isinstance(text, unicode) else text.decode('utf-8')


def encode_from_unicode(text):
    return text if isinstance(text, str) else text.encode('utf-8')


def prepare_poets():
    poets = [
        decode_to_unicode(poet) + POET_DEL
        for poet in POETRIES if len(poet) <= 100
    ]

    return poets[:1000]


def charset_from_text(text):
    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return chars, char_indices, indices_char


def build_model(history_len, hidden_num, char_num):
    """from scratch"""
    model = Sequential()
    model.add(SimpleRNN(hidden_num, return_sequences=False, input_shape=(history_len, char_num)))
    model.add(Dropout(0.2))
    model.add(Dense(char_num, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model


def build_model_from_file(struct_file, weights_file):
    model = model_from_json(open(struct_file, 'r').read())
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.load_weights(weights_file)

    return model


def save_model_to_file(model, struct_file, weights_file):
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    model.save_weights(weights_file, overwrite=True)


def build_train_data(poets, history_len, step, chars, char_indices, indices_char):
    sents = []
    next_chars = []

    for poet in poets:
        for i in range(0, len(poet) - history_len, step):
            sents.append(poet[i:i+history_len])
            next_chars.append(poet[i+history_len])

    X = np.zeros((len(sents), history_len, len(chars)), dtype=np.bool)
    Y = np.zeros((len(sents), len(chars)), dtype=np.bool)
    for i, sent in enumerate(sents):
        for t, char in enumerate(sent):
            X[i, t, char_indices[char]] = 1

        Y[i, char_indices[next_chars[i]]] = 1

    return X, Y


def train_poet():
    history_len = 10
    step = 3

    poets = prepare_poets()
    chars, char_indices, indices_char = charset_from_text(''.join(poets))
    X, Y = build_train_data(poets, history_len, step, chars, char_indices, indices_char)


    model = build_model(history_len, 512, len(chars))
    model.fit(X, Y, nb_epoch=50)

    return model

def sample(a):
    a = a / a.sum()
    return np.argmax(np.random.multinomial(1, a, 1))


def generate_poet(model, chars, char_indices, indices_char):
    seed = [''] * 9 + [POET_DEL]
    is_end = False
    length = 5
    now_length = 0
    res = u''
    while not is_end:
        if now_length > length:
            break
        x = np.zeros((1, 10, len(chars)))
        for t, char in enumerate(seed):
            if char != '':
                x[0, t, char_indices[char]] = 1

        preds = model.predict(x, verbose=0)[-1]

        next_index = sample(preds)
        next_char = indices_char[next_index]

        if now_length == length:
            if next_char in (POET_DEL, u'，', u'。'):
                now_length = -1
            else:
                while next_char not in (POET_DEL, u'，', u'。'):
                    next_index = sample(preds)
                    next_char = indices_char[next_index]
        else:
            while next_char in (POET_DEL, u'，', u'。', POET_DEL):
                next_index = sample(preds)
                next_char = indices_char[next_index]

        if next_char == POET_DEL:
            is_end = True
            break

        seed = seed[1:] + [next_char]
        res += next_char

        now_length += 1

    print res


if __name__ == '__main__':
    model = train_poet()
    save_model_to_file(model, 'epoet_struct.json', 'epoet_weights.h5')
    # model = build_model_from_file('epoet_struct.json', 'epoet_weights.h5')
    # chars, char_indices, indices_char = charset_from_text(''.join(prepare_poets()))
    # for i in range(100):
    #     sys.stdout.write('[%d] ' % i)
    #     generate_poet(model, chars, char_indices, indices_char)
