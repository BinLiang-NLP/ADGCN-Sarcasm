# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='context_indices', shuffle=True, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_context = []
        batch_context_indices = []
        batch_dependency_graph = []
        batch_sentic_graph = []
        batch_label = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            context,  context_indices,  dependency_graph, label , sentic_graph = \
                item['context'], item['context_indices'],item['dependency_graph'],item['label'],item['sentic_graph']

            context_padding = [0] * (max_len - len(context_indices))
            batch_context.append(context)
            batch_context_indices.append(context_indices + context_padding)
            batch_dependency_graph.append(numpy.pad(dependency_graph, ((0,max_len-len(context_indices)),(0,max_len-len(context_indices))), 'constant'))
            batch_sentic_graph.append(numpy.pad(sentic_graph, ((0,max_len-len(context_indices)),(0,max_len-len(context_indices))), 'constant'))
            batch_label.append(label)

        return { \
                'context': batch_context, \
                'context_indices': torch.tensor(batch_context_indices), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'sentic_graph': torch.tensor(batch_sentic_graph), \
                'label': torch.tensor( batch_label),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
