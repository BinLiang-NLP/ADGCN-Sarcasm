# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

sp_nlp = spacy.load('en_core_web_sm')


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        if embed_dim == 100:
            fname = "./glove.6B.100d.txt"
        else:
            fname = './glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['UNK'] = self.idx
            self.idx2word[self.idx] = 'UNK'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):   
        words = []

        for x in text:
            x = x.lower().strip()
            xx = sp_nlp(x)
            words = words + [str(y) for y in xx]

        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower().strip()
        words = sp_nlp(text)
        words = [str(x) for x in words]

        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class DatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = []
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 2):
                text.append(lines[i].lower().strip())
        return text

    @staticmethod
    def __read_data__(fname, tokenizer,max_seq_len = -1):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph.new', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        fin = open(fname+'.sentic', 'rb')
        idx2gragh_s = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 2):
            context = lines[i].lower().strip()
            context_indices = tokenizer.text_to_sequence(context)
            dependency_graph = idx2gragh[i]
            sentic_graph = idx2gragh_s[i]
            label = int(lines[i+1].strip())
            if len(context_indices) != dependency_graph.shape[0] or len(context_indices) != sentic_graph.shape[0]:
                print(context)
                print(len(context_indices))
                print(dependency_graph.shape,sentic_graph.shape)
                raise ValueError()
            if max_seq_len>0:
                if len(context_indices) < max_seq_len:
                    context_indices = context_indices + [0]*(max_seq_len-len(context_indices))
                else:
                    context_indices = context_indices[:max_seq_len]
                dependency_graph = np.zeros((len(context_indices),len(context_indices)))
                sentic_graph = np.zeros((len(context_indices),len(context_indices)))
            
            data = {
                'context': context,
                'context_indices': context_indices,
                'dependency_graph': dependency_graph,
                'label' : label,
                'sentic_graph': sentic_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300,max_seq_len = -1):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'iac1': {
                'train': './datasets/IAC1/train.raw',
                'test': './datasets/IAC1/test.raw'
            },
            'riloff':{
                'train': './datasets/riloff/train.raw',
                'test': './datasets/riloff/test.raw',
                'examples' : './datasets/riloff/examples.raw',
            },
            'iac2': {
                'train': './datasets/IAC2/train.raw',
                'test': './datasets/IAC2/test.raw'
            },
            'ptacek': {
                'train': './datasets/ptacek/train.raw',
                'test': './datasets/ptacek/test.raw',
                'examples' : './datasets/ptacek/examples.raw',
            },
            'movies': {
                'train': './datasets/movies/train.raw',
                'test': './datasets/movies/test.raw'
            },
            'tech': {
                'train': './datasets/tech/train.raw',
                'test': './datasets/tech/test.raw'
            },
        }


        text = DatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = Dataset(DatesetReader.__read_data__(fname[dataset]['train'], tokenizer,max_seq_len))
        self.test_data = Dataset(DatesetReader.__read_data__(fname[dataset]['test'], tokenizer,max_seq_len))
    
