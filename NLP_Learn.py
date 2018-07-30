# /usr/bin/python
# -*- encoding:utf-8 -*-

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

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
    test_tfidf()


