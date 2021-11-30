# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(document)

    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        matrix[token.i][token.i] = 1
        for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
    return matrix

def process(filename):

    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()

    idx2graph = {}

    with open(filename+'.graph.new', 'wb') as fout:
        for i in tqdm(range(0, len(lines), 2)):
            text = lines[i].lower().strip()
            adj_matrix = dependency_adj_matrix(text)
            idx2graph[i] = adj_matrix
        pickle.dump(idx2graph, fout)        


if __name__ == '__main__':

    process('./datasets/ptacek/train.raw')
    process('./datasets/ptacek/test.raw')

    process('./datasets/IAC1/train.raw')
    process('./datasets/IAC1/test.raw')

    process('./datasets/IAC2/train.raw')
    process('./datasets/IAC2/test.raw')

    process('./datasets/movies/train.raw')
    process('./datasets/movies/test.raw')

    process('./datasets/tech/train.raw')
    process('./datasets/tech/test.raw')

    process('./datasets/riloff/train.raw')
    process('./datasets/riloff/test.raw')