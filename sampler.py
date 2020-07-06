# Part of this file is derived from 
# https://github.com/kang205/SASRec/

import numpy as np
import random
from multiprocessing import Process, Queue
import pandas as pd
import os
import json
import copy
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from sklearn.metrics import roc_auc_score

def random_neg(pos, n, s, ts):
    '''
    p: positive one
    n: number of items
    s: size of samples.
    '''
    neg = set()
    for _ in range(s):
        t = np.random.randint(1, n) 
        while t in pos or t in neg or t in ts:
            t = np.random.randint(1, n) 
        neg.add(t)
    return list(neg)



def sample_function(data, train_R_dic, subgraphs_sequence_i, n_items, n_users, batch_size, max_len, neg_size, result_queue, SEED, neg_method='rand'):
	np.random.seed(SEED)
	sss = list(np.arange(data.sequences.sequences.shape[0]))
	while True:
		train_b = random.sample(sss, batch_size)  

		neg = np.zeros([batch_size, neg_size], dtype=np.int32)
		neg_dynamics = np.zeros([batch_size, neg_size], dtype=np.int32)
		for uidx in range(len(train_b)):
			u = data.sequences.user_ids[train_b[uidx]]
			time_ind = data.sequences.sequences_time[train_b[uidx]][-1]
			for ineg in range(neg_size):
				t = np.random.randint(1, n_items)
				while t in train_R_dic[u]:
					t = np.random.randint(1, n_items)
				neg[uidx][ineg] = t
				neg_dynamics[uidx][ineg] = subgraphs_sequence_i[t][time_ind]

		result_queue.put((train_b,neg,neg_dynamics))        


class Sampler(object):
    def __init__(self, data, train_dic, subgraphs_sequence_i, n_items, n_users, batch_size=128, max_len=20, neg_size=10, n_workers=10, neg_method='rand'):
        self.result_queue = Queue(maxsize=int(512))
        self.processors = []
        self.train_dic = train_dic
        self.data = data
        self.subgraphs_sequence_i = subgraphs_sequence_i
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(self.data,
                									self.train_dic,
                									self.subgraphs_sequence_i,
                                                    n_items, 
                                                    n_users,
                                                    batch_size, 
                                                    max_len, 
                                                    neg_size, 
                                                    self.result_queue, 
                                                    np.random.randint(2e9),
                                                    neg_method)))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_function_bpr(nonzero_pairs, train_R_dic, num_user, neg_sample_rate, num_item, batch_size, result_queue, SEED):

    np.random.seed(SEED)
    while True:

        train_b = random.sample(nonzero_pairs, batch_size)     
        sample4BPR = []
        for u,i in train_b:
            while True:
                bpr_neg = np.random.randint(1, num_item) 
                if bpr_neg not in train_R_dic[u]:
                    break
            sample4BPR.append((u, i, bpr_neg))
        result_queue.put((sample4BPR))


class WarpSampler_bpr(object):
    def __init__(self, train_tuples, train_dic, num_user, neg_sample_rate, num_item, batch_size=256, n_workers=1):
        self.train_tuples = train_tuples
        self.train_dic = train_dic        
        print('number of training samples', len(self.train_tuples))
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_bpr, args=(self.train_tuples,
                                                      self.train_dic,
                                                      num_user,
                                                      neg_sample_rate,
                                                      num_item,
                                                      batch_size,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

