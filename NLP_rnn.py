# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
# 实时保存模型结构、训练出来的权重、及优化器状态并调用
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

data_path = 'D:/Python/Data/Winston_Churchil.txt'
seq_length = 100

def load_data():
    raw_text = open(data_path, encoding='utf-8').read()
    raw_text = raw_text.lower()
    chars = list(set(raw_text))
    char2int = dict((c, i) for i,c in enumerate(chars))
    int2char = dict((i, c) for i,c in enumerate(chars))
    print(len(chars), len(raw_text))

    x=[]
    y=[]
    for i in range(len(raw_text) - seq_length):
        given = raw_text[i:i + 100]
        pred = raw_text[i + 100]
        x.append([char2int[char] for char in given])
        y.append(char2int[pred])
    print(x[:3])
    print(y[:3])

    n_pattern = len(x)
    n_vocab = len(chars)
    # 把x变成LSTM需要的样子
    x = np.resize(x, (n_pattern, seq_length, 1))
    x = x / float(n_vocab)
    y = np_utils.to_categorical(y)
    print(x[0])
    print(y[0])
    return x, y, n_vocab, char2int, int2char


def lstm_model(x, y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=50, batch_size=1024)
    return model


def pred_next(input_array, n_vocab, model):
    x = np.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def y2char(y, int2char):
    ind = y.argmax()
    return int2char[ind]


def string2index(raw_input, char2int):
    res = []
    for c in raw_input[len(raw_input) - seq_length:]:
        res.append(char2int[c])
    return res


def generate_article(init, char2int, int2char, n_vocab, model, round=200):
    in_string = init.lower()
    for i in range(round):
        c = y2char(pred_next(string2index(in_string, char2int), n_vocab, model), int2char)
        in_string.append(c)
    return in_string


if __name__ == '__main__':
    x, y, n_vocab, char2int, int2char = load_data()
    model = lstm_model(x, y)
    init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
    article = generate_article(init, char2int, int2char, n_vocab, model)
    print(article)
