# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/sentiwordnet.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet


def dependency_adj_matrix(text,  senticNet):
    word_list = nlp(text)
    word_list = [str(x.lemma_) for x in word_list]
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for i in range(seq_len):
        for j in range(i,seq_len):
            word_i = word_list[i]
            word_j = word_list[j]
            if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
                continue
            sentic = abs(float(senticNet[word_i] - senticNet[word_j]))
            matrix[i][j] = sentic
            matrix[j][i] = sentic

    return matrix

def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.sentic', 'wb')
    for i in tqdm(range(0, len(lines), 2)):
        text = lines[i].lower().strip()
        adj_matrix = dependency_adj_matrix(text, senticNet)
        idx2graph[i] = adj_matrix

    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close() 

if __name__ == '__main__':
    process('./datasets/tweet_riloff/train.raw')
    process('./datasets/tweet_riloff/test.raw')
