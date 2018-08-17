# /usr/bin/python
# -*- encoding:utf-8 -*-

import os
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

data_path = 'D:/Python/Data/lstm'

def load_data():
    raw_text = ''
    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            raw_text += open(os.path.join(data_path, file), errors='ignore', encoding='utf-8').read() + '\n\n'
    raw_text = raw_text.lower()
    # 分句
    sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sentensor.tokenize(raw_text)
    corpus = []
    for sent in sents:
        corpus.append(nltk.word_tokenize(sent))
    return corpus


def w2v_model(corpus):
    model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
    return model


def get_txtMat(wordlist, wv_model):
    vocab = wv_model.wv.vocab
    res = []
    for word in wordlist:
        if word in vocab:
            res.append(wv_model[word])
    return np.array(res)


def get_trainData(text_strem, seq_len):
    x = []
    y = []
    for i in range(len(text_strem) - seq_len):
        x.append(text_strem[i:i+seq_len])
        y.append(text_strem[i+seq_len])
    x = np.reshape(x, (-1, seq_len, 128))
    y = np.reshape(y, (-1, 128))
    return x, y


if __name__ == '__main__':
    corpus = load_data()
    print(len(corpus))
    wv_model = w2v_model(corpus)
    raw_input = [item for sent in corpus for item in sent]
    print(len(raw_input))
    text_strem = get_txtMat(raw_input, wv_model)
    print(text_strem.shape)
    x, y = get_trainData(text_strem[:100000], 10)
    print(x[:3])
    print(y[:3])