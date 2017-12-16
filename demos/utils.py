# coding: utf-8
from __future__ import unicode_literals

import re
import jieba
import logging
from functools import partial, reduce


jieba.setLogLevel(logging.INFO)

PUNCTS_PATTERN = re.compile(r"[.,;:!?'\"~\[\]\(\)\{\}_—。…．，；、：！？‘’“”〕《》【】〖〗（）「」～]")
SPACES_PATTERN = re.compile(r"[\r\n\t\u00a0 ]")
SENT_SEP = u'。，！？～；：.,!?:;'


def to_halfwidth(text):
    """将文本中的全角字符转换为半角字符"""
    res = ''
    for char in text:
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0

        if inside_code < 0x0020 or inside_code > 0x7e:
            res += char
        else:
            res += chr(inside_code)

    return res


def remove_punctuations(text):
    """从文本中移除标点符号"""
    return PUNCTS_PATTERN.sub(' ', text)


def unify_whitespace(text):
    """统一文本中的空白字符为空格"""
    return SPACES_PATTERN.sub(' ', text)


def remove_redundant(text, chars):
    """将字符串中连续的指定字符压缩成一个"""
    if chars == '' or text == '':
        return text

    char_set = set(chars)
    prev = ''
    result = ''
    for ch in text:
        if ch != prev or ch not in char_set:
            result += ch

        prev = ch

    return result


def clean(text):
    funcs = [
        to_halfwidth,
        remove_punctuations,
        unify_whitespace,
        partial(remove_redundant, chars=u' ')
    ]
    cleaned_text = reduce(lambda x, fn: fn(x), [text] + funcs)
    return cleaned_text


def words_tokenize(text):
    """分词"""
    return [word.strip() for word in jieba.cut(text) if len(word.strip()) > 0]


def sents_tokenize(text, puncts=SENT_SEP):
    """分句"""
    tokens = words_tokenize(text)
    sents = []

    prev = u' '
    cur_sent = []
    for tk in tokens:
        if tk not in puncts and prev in puncts:
            sents.append(cur_sent)
            cur_sent = []

        cur_sent.append(tk)
        prev = tk

    if cur_sent:
        sents.append(cur_sent)

    return sents


def shingle(sequence, length):
    if len(sequence) < length:
        return []
    else:
        return [sequence[i:i + length] for i in range(len(sequence) - length + 1)]
