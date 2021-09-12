import pickle 
import argparse
import json
import os
import time
import re
from collections import defaultdict
import csv
import gzip
import math
from tqdm import tqdm
from six.moves import xrange

import torch 
import torch.nn as nn
from torch import LongTensor, FloatTensor
import numpy as np
from gensim.models import KeyedVectors

from utils import *

class DeepCoNNRankDataset(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super(DeepCoNNRankDataset, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args['input_dir'], "meta.pkl")
        # print(type(param_path))

        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.u_docs = para['user_docs']
        self.i_docs = para['item_docs']
        self.doc_len = para["doc_len"]
        self.word_vocab = self.indexlizer._vocab

        example_path = os.path.join(self.args['input_dir'], f"{set_name}_examples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)

        meta_path = os.path.join(self.args['input_dir'], f"meta.pkl")
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        if self.set_name == "train":
            self.negN = 4
        elif self.set_name == 'valid':
            self.negN = 99
        elif self.set_name == 'test':
            ranklist_path = os.path.join(self.args['input_dir'], f"ranklist_with_gt.json")
            with open(ranklist_path, 'r') as f:
                self.ranklist = json.load(f)
            self.ranklist_keys = list(self.ranklist.keys())
            print('finish loading json file')

    def __getitem__(self, i):
        # for each review(u_docs or i_docs) [...] 
        # NOTE: not padding 
        if self.set_name == 'test':
            u_id = int(self.ranklist_keys[i])
            i_ids = self.ranklist[str(u_id)]
            for index, item in enumerate(i_ids):
                i_ids[index] = int(item)
            u_ids = [u_id]*len(i_ids)
            u_docs = [self.meta['user_docs'][u_id]]*len(i_ids)
            i_docs = []
            for item in i_ids:
                i_docs.append(self.meta['item_docs'][item])

        else:
            u_id, i_id, rating, u_doc, i_doc = self.examples[i]
            n_ids = np.random.randint(1, self.item_num, size=self.negN)
            i_ids = [i_id]
            i_docs = [i_doc]
            u_ids = [u_id]
            u_docs = [u_doc]

            for n_id in n_ids:
                i_ids.append(n_id)
                i_docs.append(self.meta['item_docs'][n_id])
                u_ids.append(u_id)
                u_docs.append(u_doc)

        return u_ids, i_ids, u_docs, i_docs

    def __len__(self):
        if self.set_name == 'test':
            return len(self.ranklist_keys)
        else:
            return len(self.examples)

    def collate_fn(self, batch):
        u_ids_batch, i_ids_batch, u_docs_batch, i_docs_batch = zip(*batch)
        
        u_ids_batch = LongTensor(u_ids_batch)
        i_ids_batch = LongTensor(i_ids_batch)
        u_docs_batch = LongTensor(u_docs_batch)
        i_docs_batch= LongTensor(i_docs_batch)
        u_docs_batch_word_masks = get_mask(u_docs_batch)
        i_docs_batch_word_masks = get_mask(i_docs_batch)

        return u_ids_batch, i_ids_batch, u_docs_batch, i_docs_batch, u_docs_batch_word_masks, i_docs_batch_word_masks