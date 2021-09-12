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
from copy import deepcopy

import torch 
import torch.nn as nn
from torch import LongTensor, FloatTensor
import numpy as np
from gensim.models import KeyedVectors

from experiment import Experiment
from deepconn import DeepCoNNppRank
from utils import *
import _tokenizer as _tokenizer
from metrics import MetronAtK
from data import DeepCoNNRankDataset
from divide_and_create_example_doc import clean_str



class DeepCoNNExperiment(Experiment):
	def __init__(self, args, dataloaders):
		super(DeepCoNNExperiment, self).__init__(args, dataloaders)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = 'cpu'
		# dataloader
		self.train_dataloader = dataloaders["train"] if dataloaders["train"] is not None else None
		self.valid_dataloader = dataloaders["valid"] if dataloaders["valid"] is not None else None
		self.test_dataloader = dataloaders["test"] if dataloaders['test'] is not None else None
		self._metron = MetronAtK(top_k=10)

		# stats
		self.train_stats = defaultdict(list)
		self.valid_stats = defaultdict(list)

		# create output path
		if dataloaders["train"] is not None:
			self.setup()
			self.build_model(args['pretrained_path']) # self.model
			self.build_optimizer() #self.optimizer
			self.build_scheduler() #self.scheduler
			self.build_loss_func() #self.loss_func

			# print
			self.print_args()
			self.print_model_stats()


	def build_scheduler(self):
		pass

	def build_model(self, pretrain_path):
		# dirty implementation
		if self.args['use_pretrain']:
			pretrain_path = os.path.join(pretrain_path, "GoogleNews-vectors-negative300.bin") 

			
			wv_from_bin = KeyedVectors.load_word2vec_format(pretrain_path, binary=True)

			word2vec = {}
			for word, vec in zip(wv_from_bin.vocab, wv_from_bin.vectors):
				word2vec[word] = vec
			
			
			_dataset = self.train_dataloader.dataset
			word_pretrained = load_pretrained_embeddings(_dataset.word_vocab, word2vec, self.args['embedding_dim'])
		else:
			_dataset  = self.train_dataloader.dataset
			word_pretrained=None

		self.model = DeepCoNNppRank(user_size=_dataset.user_num, 
				item_size=_dataset.item_num, 
				vocab_size=len(_dataset.word_vocab), 
				kernel_sizes=[3],
				hidden_dim=self.args['hidden_dim'], 
				embedding_dim=self.args['embedding_dim'],
				dropout=self.args['dropout'], 
				latent_dim=self.args['latent_dim'], 
				doc_len=_dataset.doc_len, 
				pretrained_embeddings=word_pretrained, 
				arch=self.args['arch'])

		if self.args['parallel']:
			self.model = torch.nn.DataParallel(self.model)
			self.print_write_to_log("the model is parallel training.")
		self.model.to(self.device)

	def build_optimizer(self):
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'])
		if self.args['verbose']:
			self.print_write_to_log(re.sub(r"\n", "", self.optimizer.__repr__()))
		
	def build_loss_func(self):
		self.loss_func = nn.BCELoss()

	def train_one_epoch(self, current_epoch):
		start_time = time.time()

		self.model.train()
		epoch_loss = 0

		pbar = tqdm(total=len(self.train_dataloader))
		for i, (u_ids_batch, i_ids_batch, u_docs_batch, i_docs_batch, u_docs_batch_word_masks, i_docs_batch_word_masks) in enumerate(self.train_dataloader):
			if i == 0 and current_epoch == 0:
				print("u_docs", u_docs_batch.shape, "i_docs", i_docs_batch.shape)
			self.optimizer.zero_grad()
			u_ids_batch = u_ids_batch.to(self.device) # [bz, num_item]
			i_ids_batch = i_ids_batch.to(self.device) # [bz, num_item]
			u_docs_batch = u_docs_batch.to(self.device) # [bz, num_item, seq_len]
			i_docs_batch = i_docs_batch.to(self.device) # [bz, num_item, seq_len]
			u_docs_batch_word_masks = u_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]
			i_docs_batch_word_masks = i_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]

			ground_truth = torch.zeros_like(u_ids_batch[0]).to(self.device)
			ground_truth[0] = 1
			for j in range(u_ids_batch.shape[0]):
				predicts = self.model(u_ids_batch[j], i_ids_batch[j], 
					u_docs_batch[j], i_docs_batch[j], u_docs_batch_word_masks[j], i_docs_batch_word_masks[j])
				predicts = torch.sigmoid(predicts)
				# print(predicts)
				# print(predicts.squeeze(), ground_truth)
				if j == 0:
					loss = self.loss_func(predicts.squeeze(), ground_truth.float())
				else:
					loss += self.loss_func(predicts.squeeze(), ground_truth.float())
			loss.backward()
			pbar.update(1)

			gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
			self.optimizer.step()
			if (i+1) % self.args['log_idx'] == 0 and self.args['log']:
				elpased_time = (time.time() - start_time)
				log_text = "epoch: {}/{}, step: {}/{}, loss: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
					current_epoch, self.args['epochs'],  (i+1), len(self.train_dataloader), epoch_loss, 
					self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
				)
				self.print_write_to_log(log_text)
		pbar.close()

	def valid_one_epoch(self, current_epoch):

		self.model.eval()
		test_users = []
		test_items = []
		test_scores = []
		negative_users = []
		negative_items = []
		negative_scores = []
		pbar = tqdm(total=len(self.valid_dataloader))
		for i, (u_ids_batch, i_ids_batch, u_docs_batch, i_docs_batch, u_docs_batch_word_masks, i_docs_batch_word_masks) in enumerate(self.valid_dataloader):
			if i == 0 and current_epoch == 0:
				print("u_docs", u_docs_batch.shape, "i_docs", i_docs_batch.shape)
			self.optimizer.zero_grad()
			u_ids_batch = u_ids_batch.to(self.device) # [bz, num_item]
			i_ids_batch = i_ids_batch.to(self.device) # [bz, num_item]
			u_docs_batch = u_docs_batch.to(self.device) # [bz, num_item, seq_len]
			i_docs_batch = i_docs_batch.to(self.device) # [bz, num_item, seq_len]
			u_docs_batch_word_masks = u_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]
			i_docs_batch_word_masks = i_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]

			predicts = 0
			with torch.no_grad():
				for j in range(u_ids_batch.shape[0]):
					preds = self.model(u_ids_batch[j], i_ids_batch[j], 
						u_docs_batch[j], i_docs_batch[j], u_docs_batch_word_masks[j], i_docs_batch_word_masks[j])
					if j == 0:
						predicts = preds
					else:
						predicts = torch.cat((predicts, preds), dim=1)
			if torch.cuda.is_available():
				u_ids_batch = u_ids_batch.cpu()
				i_ids_batch = i_ids_batch.cpu()
				predicts = predicts.cpu()

			u_ids_batch = u_ids_batch.permute(1, 0) #[num_item, bz]
			i_ids_batch = i_ids_batch.permute(1, 0) #[num_item, bz]
			num_item = i_ids_batch.shape[0]
			test_users.extend(u_ids_batch[0].tolist())
			test_items.extend(i_ids_batch[0].tolist())
			test_scores.extend(predicts[0].tolist())

			# print(num_item)
			for j in range(1, num_item):
				negative_users.extend(u_ids_batch[j].tolist())
				negative_items.extend(i_ids_batch[j].tolist())
				negative_scores.extend(predicts[j].tolist())
			pbar.update(1)
		pbar.close()

		self._metron.subjects = [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]
		hit_ratio, mrr, ndcg = self._metron.cal_hit_ratio(), self._metron.call_mrr(), self._metron.cal_ndcg()
		line = 'Evaluating Epoch '+str(current_epoch)+ ' HR=:'+str(hit_ratio)+' NDCG=:'+str(ndcg)
		print('[Evaluating Epoch {}] HR={:.4f}, NDCG={:.4f}'.format(current_epoch, hit_ratio, ndcg))
		return line

	def test(self):
		self.model.eval()
		pbar = tqdm(total=len(self.test_dataloader))
		ranklist = {}
		with torch.no_grad():
			for i, (u_ids_batch, i_ids_batch, u_docs_batch, i_docs_batch, u_docs_batch_word_masks, i_docs_batch_word_masks) in enumerate(self.test_dataloader):
				if i == 0:
					print("u_docs", u_docs_batch.shape, "i_docs", i_docs_batch.shape)
				# self.optimizer.zero_grad()
				u_ids_batch = u_ids_batch.to(self.device) # [bz, num_item]
				i_ids_batch = i_ids_batch.to(self.device) # [bz, num_item]
				u_docs_batch = u_docs_batch.to(self.device) # [bz, num_item, seq_len]
				i_docs_batch = i_docs_batch.to(self.device) # [bz, num_item, seq_len]
				u_docs_batch_word_masks = u_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]
				i_docs_batch_word_masks = i_docs_batch_word_masks.to(self.device) # [bz, num_item, seq_len]

				with torch.no_grad():
					for j in range(u_ids_batch.shape[0]):
						preds = self.model(u_ids_batch[j], i_ids_batch[j], 
							u_docs_batch[j], i_docs_batch[j], u_docs_batch_word_masks[j], i_docs_batch_word_masks[j])
						test_items = i_ids_batch[j].squeeze().tolist()
						preds = preds.squeeze().tolist()
						sorted_items = []
						sorted_scores_index = sorted(range(len(preds)), key=lambda k: preds[k], reverse=True)[:100]
						for k in xrange(len(sorted_scores_index)):
							sorted_items.append(str(test_items[sorted_scores_index[k]]))
						ranklist[str(u_ids_batch[j][0].item())] = sorted_items

				pbar.update(1)
			pbar.close()
		with open(os.path.join(self.args['input_dir'],'ranklist_rerank.json'), 'w') as fp:
			json.dump(ranklist, fp)

	def train(self):
		print("start training ...")
		with open('deepconn_log.txt', 'w') as f:
			for epoch in range(self.args['epochs']):
				self.train_one_epoch(epoch)
				if self.args['evaluate']:
					if epoch % 1 == 0:
						line=self.valid_one_epoch(epoch)+'\n'
						f.write(line)