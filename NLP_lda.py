# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import re
import os
import gensim
from gensim import models, similarities, corpora

data_path = 'D:/Python/Data/HillaryEmails.csv'
model_path = 'D:/Python/Data/'

stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']


def read_data():
    df = pd.read_csv(data_path)
    df = df[['Id', 'ExtractedBodyText']].dropna()
    return df

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'\d+/\d+/\d+', '', text)
    text = re.sub(r'[0-2]?[0-9]:[0-6][0-9]', '', text)
    text = re.sub(r'[\w]+@[\.\w]+]', '', text)
    text = re.sub(r'https?://[\w\-]+\.+[\w\-\./%&=\?]+', '', text)
    # 剔除所有其他字符
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join([word for word in pure_text.split() if len(word)>1])
    text = ' '.join([word for word in pure_text.split() if len(word)>1])
    return text

def make_corpus(docs):
    doclist = [clean_text(doc) for doc in docs['ExtractedBodyText'].values]
    texts = []
    for doc in doclist:
        line = [word for word in doc.lower().split() if word not in stoplist]
        if len(line) > 2:
            texts.append(line)
    return texts


class LDA_model():

    def __init__(self, num_topics=20, tfidf=False):
        self.num_topics = num_topics
        self.tfidf = tfidf

    def train_model(self):
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        if self.tfidf:
            corpus = models.TfidfModel(corpus)[]
        self.lda = models.LdaModel(corpus=corpus, id2word=self.dictionary,
                                   num_topics=self.num_topics, minimum_probability=0.001)

    def update_model(self, texts):
        other_corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda.update(other_corpus)

    def save_model(self, save_path):
        self.lda.save(os.path.join(save_path, 'lda.model'))
        self.dictionary.save(os.path.join(save_path, 'lda.dict'))

    def load_model(self, save_path):
        self.lda = models.LdaModel.load(os.path.join(save_path, 'lda.model'))
        self.dictionary = corpora.Dictionary.load(os.path.join(save_path, 'lda.dict'))
        self.num_topics = self.lda.num_topics

    def get_topic(self, topn=10):
        topic_list = []
        for i in range(self.num_topics):
            word_list = self.lda.get_topic_terms(topicid=i, topn=topn)
            topic_list.append([(self.dictionary.id2token[key], prob) for key, prob in word_list])
        return topic_list

    def get_doc_topic(self, doc):
        doclist = self.dictionary.doc2bow(doc)
        # return self.lda.get_document_topics(doclist)
        return self.lda[doclist]

if __name__ == '__main__':
    docs = read_data()
    texts = make_corpus(docs)
    texts_train = texts[:4000]
    texts_update = texts[4000:]
    lda = LDA_model()
    lda.train_model(texts_train, 10)
    lda.save_model(model_path)
    lda.load_model(model_path)
    lda.update_model(texts_update)

    for i in range(10):
        print(texts_update[i])
        print(lda.get_doc_topic(texts_update[i]))
