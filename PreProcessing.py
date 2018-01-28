import csv
from nltk import wordpunct_tokenize, PorterStemmer, LancasterStemmer
import os
import pandas as pd
import numpy as np

data = pd.read_csv('\\train\\train.txt', sep="	", header=None, names=["class","text"])[1:]

stopWordsFile = open("\\IR-Project\\StopWord.txt", 'r')
stopWords = stopWordsFile.read().split('\n')
porter = PorterStemmer()
import re

d = []
def train():
    for item in data['text']:
        item = item.strip().lower()
        item= re.sub(r'http\S+', '', item)
        s=[]
        tokens = wordpunct_tokenize(item)
        stopWordsRemoved = [token for token in tokens if token not in stopWords and len(token)>=2 and not token.isdigit()]
        stems = [porter.stem(word) for word in stopWordsRemoved]
        print(stems)
        d.append(stems)

    with open("phase1.csv", 'w') as myfile:
        wr = csv.writer(myfile,  lineterminator='\n')
        for i in d:
            wr.writerow(i)
    print("train completed")

train()