import torch
from torch.autograd import Variable

import sys
from tqdm import tqdm
import time
import json
import numpy as np
from six.moves import xrange
import heapq
import os

from utils import *
from gmf import *

class Engine(object):
	# Meta Engine
	def __init__(self, config):
		self.config = config  # model configuration
		self._metron = MetronAtK(top_k=10)
		self.opt = use_optimizer(self.model, config)
		# self.crit = torch.nn.MSELoss()
		self.crit = torch.nn.BCEWithLogitsLoss()

	def train_single_batch(self, users, items, ratings):
		assert hasattr(self, 'model'), 'Please specify the exact model !'
		if self.config['use_cuda'] is True:
			users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
		self.opt.zero_grad()
		ratings_pred = self.model(users, items)
		# print(ratings.shape)
		loss = self.crit(ratings_pred.view(-1), ratings)
		loss.backward()
		self.opt.step()
		loss = loss.item()
		return loss

	def train_an_epoch(self, train_loader, epoch_id):
		assert hasattr(self, 'model'), 'Please specify the exact model !'
		if self.config['use_cuda'] is True:
			self.model.cuda()
		self.model.train()
		total_loss = 0
		pbar = tqdm(total=len(train_loader))
		for batch_id, batch in enumerate(train_loader):
			assert isinstance(batch[0], torch.LongTensor)
			user, item, rating = batch[0], batch[1], batch[2]
			rating = rating.float()
			loss = self.train_single_batch(user, item, rating)
			gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
			# if batch_id % 1000==0:
			# 	print('gradient norm: ', gnorm)
			total_loss += loss
			pbar.update(1)
		pbar.close()
		print('Epoch Loss: ', total_loss)

	def evaluate(self, evaluate_data, epoch_id):
		assert hasattr(self, 'model'), 'Please specify the exact model !'
		self.model.eval()
		self.model.cpu()
		with torch.no_grad():
			test_users, test_items = evaluate_data[0], evaluate_data[1]
			negative_users, negative_items = evaluate_data[2], evaluate_data[3]
			test_users = test_users.cpu()
			test_items = test_items.cpu()
			test_scores = self.model(test_users, test_items)
			negative_users = negative_users.cpu()
			negative_items = negative_items.cpu()
			negative_scores = self.model(negative_users, negative_items)
			negative_scores = negative_scores.cpu()
			self._metron.subjects = [test_users.data.view(-1).tolist(),
								 test_items.data.view(-1).tolist(),
								 test_scores.data.view(-1).tolist(),
								 negative_users.data.view(-1).tolist(),
								 negative_items.data.view(-1).tolist(),
								 negative_scores.data.view(-1).tolist()]
		hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
		print('[Evaluating Epoch {}] HR={:.4f}, NDCG={:.4f}'.format(epoch_id, hit_ratio, ndcg))
		return hit_ratio, ndcg

	def output_ranklist_large(self, test_loader, interact_hist, item_pool, output_dir, mod='rank'):
		assert hasattr(self, 'model'), 'Please specify the exact model !'
		self.model.eval()
		self.model.cpu()
		print('Outputting Ranklist')
		if mod == 'rank':
			with torch.no_grad():
				full_ranklist = {}
				user_embedding = self.model.embedding_user.weight.cpu()
				item_embedding = self.model.embedding_item.weight.data.cpu().transpose(1,0)
				pbar = tqdm(total=len(test_loader))
				test_items = [k for k in range(item_embedding.shape[1])]
				# test_items = torch.tensor(test_items)
				for i, batch in enumerate(test_loader):
					test_user = batch
					user_embedding_single = user_embedding[test_user].data
					element_product = torch.matmul(user_embedding_single, item_embedding).cpu().squeeze()
					for j in range(batch.shape[0]):
						test_user_single = test_user[j].item()
						sorted_items = []
						test_scores = element_product[j].tolist()
						sorted_scores_index = sorted(range(len(test_scores)), key=lambda k: test_scores[k], reverse=True)[:1000]
						for k in xrange(len(sorted_scores_index)):
							sorted_items.append(str(test_items[sorted_scores_index[k]]))
						full_ranklist[str(test_user_single)] = sorted_items
					pbar.update(1)
				pbar.close()
			with open(os.path.join(output_dir,'ranklist.json'), 'w') as fp:
				json.dump(full_ranklist, fp)
		elif mod == 'rerank':
			with torch.no_grad():
				full_ranklist = {}
				user_embed = self.model.embedding_user.weight.cpu()
				item_embed = self.model.embedding_item.weight.cpu()
				pbar = tqdm(total=len(test_loader))
				for i, batch in enumerate(test_loader):
					test_users = batch[0]
					test_items = batch[1]
					user_embed_batch = user_embed[test_users][:][0]
					item_embed_batch = item_embed[test_items].transpose(1, 0)
					element_product = torch.matmul(user_embed_batch, item_embed_batch)
					sorted_items = []
					test_items = test_items.tolist()
					test_users = test_users.tolist()
					test_scores = element_product.tolist()
					sorted_scores_index = sorted(range(len(test_scores)), key=lambda k: test_scores[k], reverse=True)
					for j in xrange(len(sorted_scores_index)):
						sorted_items.append(str(test_items[sorted_scores_index[j]]))
					full_ranklist[str(test_users[0])] = sorted_items
					pbar.update(1)
				pbar.close()
			with open(os.path.join(output_dir,'ranklist_rerank.json'), 'w') as fp:
				json.dump(full_ranklist, fp)

	# def output_ranklist(self, test_loader, interact_hist, item_pool, output_dir):
	# 	assert hasattr(self, 'model'), 'Please specify the exact model !'
	# 	self.model.eval()
	# 	self.model.cpu()
	# 	print('Outputting Ranklist')

	# 	with torch.no_grad():
	# 		full_ranklist = {}
	# 		user_embedding = self.model.embedding_user.weight.cpu()
	# 		item_embedding = self.model.embedding_item.weight.data.cpu().transpose(1,0)
	# 		pbar = tqdm(total=len(test_loader))
	# 		# test_items = [k for k in range(item_embedding.shape[1])]
	# 		for i, batch in enumerate(test_loader):
	# 			test_user = batch
	# 			user_embedding_single = user_embedding[test_user].data
	# 			element_product = torch.matmul(user_embedding_single, item_embedding).cpu().squeeze()
	# 			for j in range(batch.shape[0]):
	# 				test_user_single = test_user[j].item()
	# 				test_items = list(item_pool-set(interact_hist[test_user_single]))
	# 				# test_items = list(item_pool)
	# 				test_scores = element_product[j][test_items].tolist()
	# 				sorted_items = []
	# 				# test_scores = element_product[j].tolist()
	# 				sorted_scores_index = sorted(range(len(test_scores)), key=lambda k: test_scores[k], reverse=True)[:1000]
	# 				for k in xrange(len(sorted_scores_index)):
	# 					sorted_items.append(str(test_items[sorted_scores_index[k]]))
	# 				full_ranklist[str(test_user_single)] = sorted_items
	# 			pbar.update(1)
	# 		pbar.close()
	# 	with open(output_dir+'ranklist_test.json', 'w') as fp:
	# 		json.dump(full_ranklist, fp)


def get_top_k_index(score_list, item_list, k=1000):
	top_k = heapq.nlargest(k, score_list)
	sorted_items = []
	for item in top_k:
		sorted_items.append(item_list[score_list.index(item)])
	return sorted_items

def heapsort(iterable):
	h = []
	for value in iterable:
		heapq.heappush(h, value)
	return [heapq.heappop(h) for i in range(len(h))]

if __name__ == '__main__':
	scores = list(np.random.rand(2000000))
	time0 = time.time()
	# print(scores)
	sorted_scores_index=sorted(scores)[:1000]
	time1 = time.time()
	print(time1-time0)
	sorted_scores_index=heapq.nlargest(1000, scores)
	print(time.time()-time1)


