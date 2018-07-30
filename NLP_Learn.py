# /usr/bin/python
# -*- encoding:utf-8 -*-

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import re
import sys
import time
from collections import Counter
import string
import jieba
import jieba.analyse as analyse
import jieba.posseg as pseg


def test_strip():
    s = ' hello, word '
    print(s.strip())
    print(s.lstrip())
    print(s.rstrip())

def test_index():
    s = 'hello,word'
    print(s.index('h'))
    print(s.index('wo'))
    # index()查找不到时报异常
    # print(s.index('a'))
    print(s.find('h'))
    print(s.find('wo'))
    print(s.find('a'))

def test_up_low():
    s = 'Hello'
    print(s.upper())
    print(s.lower())

def test_reverse():
    s = 'abcde'
    print(s[::-1])

def get_max_value_v1(text):
    text = text.lower()
    result = re.findall('[a-zA-Z]', text)
    count = Counter(result)
    count_list = list(count.values())
    max_value = max(count_list)
    max_list = []
    # print(count.items())
    for k, v in count.items():
        if v == max_value:
            max_list.append(k)
    max_list = sorted(max_list)
    print(max_list)

def get_max_value_v2(text):
    count = Counter([x for x in text if x.isalpha()])
    m = max(count.values())
    max_list = sorted([x for (x, y) in count.items() if y == m])
    print(max_list)

def get_max_value_v3(text):
    text = text.lower()
    # max(iterable, key, default) 求迭代器的最大值，其中iterable 为迭代器，
    # max会for i in … 遍历一遍这个迭代器，然后将迭代器的每一个返回值当做参数
    # 传给key = func，然后将func的执行结果传给key，然后以key为标准进行大小的判断
    print(max(string.ascii_lowercase, key=text.count))

def test_re_1():
    # 一个()代表一个组, 可以用\1, \2访问各组
    pattern = re.compile(r'(\w+) (\w+)(?P<sign>.*)')
    match = pattern.match('hello yaolei!')
    if match:
        print('match.string: ', match.string)
        print('match.re: ', match.re)
        print('match.pos: ', match.pos)
        print('match.endpos: ', match.endpos)
        print('match.lastindex: ', match.lastindex)
        print('match.lastgroup: ', match.lastgroup)

        print('match.group(1, 2): ', match.group(1, 2))
        print('match.groups(): ', match.groups())
        print('match.groupdict(): ', match.groupdict())
        print('match.expand(\'\\2 \\1\\3\'): ', match.expand(r'\2 \1\3'))

def test_re_2():
    pattern = re.compile(r'\d+')
    s = 'one1two22three333four444five5555'
    print(pattern.split(s))
    print(pattern.findall(s))


def test_re_3():
    p = re.compile(r'(\w+) (\w+)')
    s = 'i say, hello yaolei!'
    print(p.sub(r'\2 \1', s))
    print(p.subn(r'\2 \1', s))

    def func(m):
        return m.group(1) + ' ' + m.group(2)
    print(p.sub(func, s))

def test_jieba():
    s = '我在学习自然语言处理'
    cut_all_true = '|'.join(jieba.cut(s, cut_all=True))
    cut_all_false = '|'.join(jieba.cut(s, cut_all=False))
    cut_all_default = '|'.join(jieba.cut(s))
    cut_list = jieba.lcut(s)
    print(cut_all_true)
    print(cut_all_false)
    print(cut_all_default)
    print(cut_list)
    # jieba.load_userdict(path) 添加用户自定义词典
    # 用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来
    print(jieba.suggest_freq(('学', '习'), True))
    print('|'.join(jieba.cut(s)))

    # 并行分词, 仅支持Linux
    # jieba.enable_parallel()
    # content = open('D:\\Python\\Data\\西游记.txt').read()
    # t1 = time.time()
    # words = jieba.cut(content)
    # t2 = time.time()
    # print(t2-t1)
    #
    # jieba.disable_parallel()
    # content = open('D:\\Python\\Data\\西游记.txt').read()
    # t1 = time.time()
    # words = jieba.cut(content)
    # t2 = time.time()
    # print(t2 - t1)

def test_tfidf():
    lines = open('D:\\Python\\Data\\NBA.txt', encoding='utf-8').read()
    print(type(lines))
    # 基于TF-IDF算法的关键词抽取
    words = analyse.extract_tags(lines, topK=20, withWeight=True, allowPOS=())
    print(words)

    # 基于TextRank算法的关键词抽取
    words = analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    print(words)
    words = analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n'))
    print(words)

    # 词性标注
    words = pseg.cut('我爱自然语言处理')
    # print(list(words))
    for word, flag in words:
        print(word, flag)

    # Tokenize：返回词语在原文的起止位置
    result = jieba.tokenize('我爱自然语言处理')
    print(list(result))


def test_tfidf():
    corpus = ["我 来到 北京 清华大学",
              "他 来到 了 网易 杭研 大厦",
              "小明 硕士 毕业 与 中国 科学院",
              "我 爱 北京 天安门"]
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    vect = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(vect)
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print(word)
    print(weight)


if __name__ == '__main__':
    # test_strip()
    # test_index()
    # test_up_low()
    # test_reverse()
    # get_max_value_v1('Hello, word')
    # get_max_value_v2('Hello, word')
    # get_max_value_v3('Hello, word')
    # test_re_1()
    # test_re_2()
    # test_re_3()
    test_jieba()
    test_tfidf()
    test_tfidf()


