# /usr/bin/python
# -*- encoding:utf-8 -*-

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

