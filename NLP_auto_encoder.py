# /usr/bin/python
# -*- encoding:utf-8 -*-

from keras.layers import Dense, Input
from keras.models import Model
from sklearn.cluster import KMeans

class ASCIIautoencoder():
    # 基于字符的autoencoder
    def __init__(self, sen_len=512, encodeing_dim=32, epoch=50, val_ratio=0.3):
        self.sen_len = sen_len
        self.encoding_dim = encodeing_dim
        self.epoch = epoch
        self.autoencoder = None
        self.decoder = None
        self.kmeanmodel = KMeans(n_clusters=2)

    def fit(self, x):
        x_train = self.preprocess(x, length=self.sen_len)

        input_text = Input(shape=(self.sen_len,))
        encoded = Dense(1024, activation='tanh')(input_text)
        encoded = Dense(512, activation='tanh')(encoded)
        encoded = Dense(128, activation='tanh')(encoded)
        encoded = Dense(self.encoding_dim, activation='tanh')(encoded)

        decoded = Dense(128, activation='tanh')(encoded)
        decoded = Dense(512, activation='tanh')(decoded)
        decoded = Dense(1024, activation='tanh')(decoded)
        decoded = Dense(self.sen_len, activation='sigmoid')(decoded)

        self.autoencoder = Model(inputs=input_text, outputs=decoded)
        self.decoder = Model(inputs=input_text, outputs=encoded)



    def preprocess(self, s_list, length=256):
         pass
