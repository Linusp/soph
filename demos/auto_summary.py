# coding: utf-8
from __future__ import absolute_import, print_function

import sys
import cPickle
from itertools import chain
from collections import Counter

import numpy as np

from nlp.utils import (
    clean,
    sents_tokenize,
)


def simple_auto_summary(text, stopwords=None, number=5):
    """"""
    # 分句并去除标点
    sents = []
    for sent in sents_tokenize(text):
        cur_sent = filter(lambda x: x != u'', [clean(w).strip() for w in sent])
        if cur_sent:
            sents.append(cur_sent)

    words = set(chain.from_iterable(sents))
    term_freqs = Counter(words)
    stopwords = stopwords or set()
    for word in words & stopwords:
        term_freqs.pop(word)

    sents_score = Counter()
    for i, sent in enumerate(sents):
        score = sum(term_freqs[word] for word in sent)
        sents_score[i] = score

    return [sents[i] for i, s in sorted(sents_score.most_common(number), key=lambda x: x[0])]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} <text> <stopfile>'.format(sys.argv[0]))
        sys.exit(1)

    text = open(sys.argv[1]).read().strip()
    stop = cPickle.load(open(sys.argv[2], 'r'))

    for sent in simple_auto_summary(text, stop, 5):
        print(''.join(sent))
