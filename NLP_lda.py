# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import re


data_path = 'D:/Python/Data/HillaryEmails.csv'

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
    text = ' '.join([word for word in pure_text.split()])
    return text


if __name__ == '__main__':
    docs = read_data()
    doclist = [clean_text(doc) for doc in docs['ExtractedBodyText'].values]
    print(doclist[:3])
