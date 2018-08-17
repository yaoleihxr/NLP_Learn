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
seq_len = 10

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


def get_trainData(text_strem):
    x = []
    y = []
    for i in range(len(text_strem) - seq_len):
        x.append(text_strem[i:i+seq_len])
        y.append(text_strem[i+seq_len])
    x = np.reshape(x, (-1, seq_len, 128))
    y = np.reshape(y, (-1, 128))
    return x, y


def lstm_model(x, y):
    model = Sequential()
    model.add(LSTM(units=256, dropout=0.2, recurrent_dropout=0.2, input_shape=(seq_len, 128)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=50, batch_size=1024)
    return model


def str2index(raw_input, wv_model):
    raw_input = raw_input.lower
    input_strem = nltk.word_tokenize(raw_input)
    vocab = wv_model.wv.vocab
    word_strem = [word for word in input_strem if word in vocab]
    res = []
    for word in word_strem[len(word_strem) - seq_len:]:
        res.append(wv_model[word])
    return res


def predict_next(input_array, lstmmodel):
    x = np.reshape(input_array, (-1, seq_len, 128))
    y = lstmmodel.predict(x)
    return y


def y2word(y, wv_model):
    word = wv_model.most_similar(positive=y, topn=1)
    return word


def generate_article(init, wv_model, lstmmodel, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        word = y2word(predict_next(str2index(in_string, wv_model),lstmmodel),wv_model)
        in_string += ' ' + word[0][0]
    return in_string


if __name__ == '__main__':
    corpus = load_data()
    print(len(corpus))
    wv_model = w2v_model(corpus)
    raw_input = [item for sent in corpus for item in sent]
    print(len(raw_input))
    text_strem = get_txtMat(raw_input, wv_model)
    print(text_strem.shape)
    x, y = get_trainData(text_strem[:100000])
    model = lstm_model(x, y)

    init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
    article = generate_article(init)
    print(article)

