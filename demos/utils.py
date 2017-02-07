# coding: utf-8

import re
import jieba
import logging
from functools import partial


jieba.setLogLevel(logging.INFO)

PUNCTS_PATTERN = re.compile(ur"[.,;:!?'\"~\[\]\(\)\{\}_—。…．，；、：！？‘’“”〕《》【】〖〗（）「」～]")
SPACES_PATTERN = re.compile(ur"[\r\n\t\u00a0 ]")
SENT_SEP = u'。，！？～；：.,!?:;'


def encode_from_unicode(text):
    """将文本转换为 str 格式"""
    return text.encode('utf-8') if isinstance(text, unicode) else text


def decode_to_unicode(text):
    """将文本转换为 unicode 格式"""
    return text.decode('utf-8') if isinstance(text, str) else text


def to_halfwidth(text):
    """将文本中的全角字符转换为半角字符"""
    text = decode_to_unicode(text)

    res = u''
    for uchar in text:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0

        if inside_code < 0x0020 or inside_code > 0x7e:
            res += uchar
        else:
            res += unichr(inside_code)

    return res


def remove_punctuations(text):
    """从文本中移除标点符号"""
    text = decode_to_unicode(text)
    return PUNCTS_PATTERN.sub(u' ', text)


def unify_whitespace(text):
    """统一文本中的空白字符为空格"""
    text = decode_to_unicode(text)
    return SPACES_PATTERN.sub(u' ', text)


def remove_redundant(text, chars):
    """将字符串中连续的指定字符压缩成一个"""
    text = decode_to_unicode(text)
    chars = decode_to_unicode(chars)
    if chars == u'' or text == u'':
        return text

    char_set = set(chars)
    prev = u''
    result = u''
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
    text = decode_to_unicode(text)
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
        return [sequence[i:i+length] for i in xrange(len(sequence) - length + 1)]
