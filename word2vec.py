import numpy as np
import pandas as pd
import MeCab


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans

import json
import requests
import xmltodict
import gensim
from pprint import pprint
# from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(file_path):
    return pd.read_csv(file_path)

def load_file(file_path):
    with open(file_path) as f:
        return f.readlines()

def get_mecab_wakati():
    return MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

def get_mecab_morpheme():
    return MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

def w2c(document):
    # def rep(sentence):
    #     sentence = sentence.replace("。", "")
    #     sentence = sentence.replace("「", "")
    #     sentence = sentence.replace("」", "")
    #     sentence = sentence.replace("」", "")
    #     sentence = sentence.replace("（", "")
    #     sentence = sentence.replace("）", "")
    #     sentence = sentence.replace("(", "")
    #     sentence = sentence.replace(")", "")

    # print("文章を入力しました")
    print(document)
    if "。" in document:
        document = [sentence.replace("\r\n", "") for sentence in document.split("。") if sentence is not ""]
    else:
        document = [sentence for sentence in document.split("\r\n") if sentence is not ""]
    mecab = get_mecab_wakati()
    mecab.parse("")
    copus = [mecab.parse(sentence).strip().split(' ') for sentence in document]
    print(copus)
    # print("文章を分かち書きしました")
    # print(copus)
    model = gensim.models.Word2Vec(copus, size=100,min_count=1,window=5,iter=1000)
    print(model.wv.vocab)
    return model

def word_in_document(document):
    if "。" in document:
        document = [sentence.replace("\r\n", "") for sentence in document.split("。") if sentence is not ""]
    else:
        document = [sentence for sentence in document.split("\r\n") if sentence is not ""]
    mecab = get_mecab_morpheme()
    mecab.parse("")
    words = []
    for sentence in document:
        node = mecab.parseToNode(sentence)
        while node:
            if node.feature.split(",")[0] == '名詞':
                words.append(node.surface)
            node = node.next
    return sorted(set(words), key=words.index)



def fcm(n_clusters, vector):
    cntr, u, u0, d, jm, p, fpc = cmeans(vector.T, n_clusters, 2.5, 0.0001, 1000, seed=0)
    return {
        "center": cntr,
        "membership": u,
        "label": u.argmax(axis=0)
    }

if __name__ == "__main__":
    document = load_file("sentence.txt")
    mecab = get_mecab()
    mecab.parse("")
    copus = [mecab.parse(sentence).strip().split(' ') for sentence in document]
    print(len(copus))
    print(copus)

    model = gensim.models.Word2Vec(copus, size=2,min_count=1,window=5,iter=1000)
    # model = gensim.models.FastText(copus, size=100, min_count=10, window=5, iter=100)

    wordvector = []

    for word in model.wv.vocab.keys():
        wordvector.append(model.wv[word])
    print(model.wv.vectors)

    wordvector = np.array(wordvector)
    
    fuzzy_cmeans = cmeans(wordvector.T, 10, 2.5, 0.0001, 1000)
    cntr, u, u0, d, jm, p, fpc = fuzzy_cmeans

    print("クラスタ数 {}".format(10))
    print("クラスタ中心 {}".format(cntr))

    print("クラスタメンバーシップ")
    pprint(u)
    print("クラスタ割当")
    pprint(u.argmax(axis=0))

    # kmeans = KMeans(n_clusters=10)
    # kmeans.fit(wordvector)
    
    # print("クラスタ数 {}".format(kmeans.n_clusters))
    # print(kmeans.labels_)


    
