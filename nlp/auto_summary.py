# coding: utf-8

from __future__ import absolute_import

from itertools import chain
from collections import Counter

import numpy as np

from nlp.utils import (
    clean,
    sents_tokenize,
)


def simple_auto_summary(text, stopwords):
    # 分句并去除标点
    sents = []
    for sent in sents_tokenize(text):
        cur_sent = filter(lambda x: x != u'', [clean(w).strip() for w in sent])
        if cur_sent:
            sents.append(cur_sent)

    words = list(chain.from_iterable(sents))
    term_freqs = Counter(words)
