# /usr/bin/python
# -*- encoding:utf-8 -*-

import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

data_path = 'D:/Python/Data/Combined_News_DJIA.csv'
stop = stopwords.words('english')
# 提取词干
wordnet_lemmatizer = WordNetLemmatizer()


# 去除停止词、数字、符号
def check_words(word):
    word = word.lower()
    if word in stop:
        return False
    elif re.search(r'\d', word):
        return False
    elif re.match(r'\W', word):
        return False
    else:
        return True


# 句子预处理
def preprocess(sent):
    res = []
    for word in sent:
        if check_words(word):
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


# 载入数据
def data_load():
    data = pd.read_csv(data_path)
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']

    x_train = train.values[:, 2:].astype(str)
    x_test = test.values[:, 2:].astype(str)
    corpus = x_train.flatten()

    x_train = np.array([' '.join(x) for x in x_train])
    x_test = np.array([' '.join(x) for x in x_test])
    y_train = train['Label'].values
    y_test = test['Label'].values

    x_train = [word_tokenize(x) for x in x_train]
    x_test = [word_tokenize(x) for x in x_test]
    corpus = [word_tokenize(x) for x in corpus]

    x_train = [preprocess(x) for x in x_train]
    x_test = [preprocess(x) for x in x_test]
    corpus = [preprocess(x) for x in corpus]
    return x_train, x_test, y_train, y_test, corpus


# 训练word2vec
def train_word2vec(corpus):
    model = Word2Vec(corpus, size=128, window=5, min_count=5)
    return model


# 计算句子向量，取平均
def get_vector_avg(word_list, model):
    vocab = model.wv.vocab
    ret_vec = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            ret_vec += model[word]
            count += 1
    if count > 0:
        ret_vec = ret_vec / count
    return ret_vec


def svm_model():
    x_train, x_test, y_train, y_test, corpus = data_load()
    wv_model = train_word2vec(corpus)
    x_train = [get_vector_avg(x, wv_model) for x in x_train]
    x_test = [get_vector_avg(x, wv_model) for x in x_test]

    params = [0.1, 0.5, 1, 3, 5, 7, 10, 12, 16, 20, 25, 30, 35, 40]
    test_scores = []
    for param in params:
        clf = SVR(gamma=param)
        test_score = cross_val_score(clf, x_train, y_train, cv=3, scoring='roc_auc')
        test_scores.append(np.mean(test_score))
    print(test_scores)


# CNN solution #
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# 将句子转化为word2vec矩阵
def trans_to_matrix(x, model, padding=256, vec_size=128):
    res = []
    for sent in x:
        matrix = []
        for i in range(padding):
            try:
                matrix.append(list(model[sent[i]]))
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res

def CNN_model():
    x_train, x_test, y_train, y_test, corpus = data_load()
    wv_model = train_word2vec(corpus)
    x_train = np.array(trans_to_matrix(x_train, wv_model))
    x_test = np.array(trans_to_matrix(x_test, wv_model))
    print(x_train.shape)
    print(x_test.shape)



if __name__ == '__main__':
    # svm_model()
    CNN_model()
