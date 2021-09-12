import random, sys, time
import pandas as pd
from copy import deepcopy
import numpy as np
import json
from tqdm import tqdm
import math
from six.moves import xrange
import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

random.seed(0)

class UserDataset(Dataset):
	"""Wrapper, convert <user> Tensor into Pytorch Dataset"""
	def __init__(self, user_tensor):
		"""
		args:
			target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
		"""
		self.user_tensor = user_tensor

	def __getitem__(self, index):
		return self.user_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)

class UserItemDataset(Dataset):
	# wrapper, convert <user, item> tensor into pytorch dataset
	def __init__(self, user_tensor, item_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)


class UserItemRatingDataset(Dataset):
	"""Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
	def __init__(self, user_tensor, item_tensor, target_tensor):
		"""
		args:

			target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
		"""
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.target_tensor = target_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)


class SampleGenerator(object):
	def __init__(self, ratings, input_dir):
		ratings.columns=['userId', 'itemId', 'rating', 'timestamp']
		self.ratings = ratings.astype(int)
		self.user_pool = set(self.ratings['userId'].unique())
		self.item_pool = set(self.ratings['itemId'].unique())
		self.data_dir = input_dir

	def _normalize(self, ratings):
		"""normalize into [0, 1] from [0, max_rating], explicit feedback"""
		ratings = deepcopy(ratings)
		max_rating = ratings.rating.max()
		ratings['rating'] = ratings.rating * 1.0 / max_rating
		return ratings
	
	def _binarize(self, ratings):
		"""binarize into 0 or 1, imlicit feedback"""
		ratings = deepcopy(ratings)
		ratings['rating'][ratings['rating'] > 0] = 1.0
		return ratings

	def _split(self, ratings, method='ufo', test_size=0.3):
		# method = [''tloo', 'utfo', 'ufo']
		# tloo: leave one out with timestamp
		# utfo: time-aware split by ratio in user level
		# ufo: user level ratio split
		trainset = pd.DataFrame()
		testset = pd.DataFrame()
		if method == 'tloo':
			ratings = ratings.sort_values(['timestamp']).reset_index(drop=True)
			ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
			trainset, testset = ratings[ratings['rank_latest'] > 1].copy(), ratings[ratings['rank_latest']==1].copy()
			del trainset['rank_latest'], testset['rank_latest']

		if method == 'loo':
			test_index = ratings.groupby(['userId']).apply(lambda grp: np.random.choice(grp.index))
			testset = ratings.loc[test_index, :].copy()
			trainset = ratings[~ratings.index.isin(test_index)].copy()

		elif method == 'ufo':
			ratings = ratings.sort_values(['userId']).reset_index(drop=True)
			def time_split(grp):
				start_idx = grp.index[0]
				split_len = int(np.ceil(len(grp) * (1 - test_size)))
				split_idx = start_idx + split_len
				end_idx = grp.index[-1]
				if len(grp) <= 2:
					return [start_idx]
				else:
					return list(range(start_idx, split_idx))
			# middle = ratings.groupby('userId').apply(time_split)
			train_idx = ratings.groupby('userId').apply(time_split).explode().values
			trainset = ratings.loc[train_idx, :]
			testset = ratings[~ratings.index.isin(train_idx)]

		elif method == 'utfo':
			ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)
			def time_split(grp):
				start_idx = grp.index[0]
				split_len = int(np.ceil(len(grp) * (1 - test_size)))
				split_idx = start_idx + split_len
				end_idx = grp.index[-1]
				if len(grp) <= 2:
					return [start_idx]
				else:
					return list(range(start_idx, split_idx))
			middle = ratings.groupby('userId').apply(time_split)
			train_idx = ratings.groupby('userId').apply(time_split).explode().values
			trainset = ratings.loc[train_idx, :]
			testset = ratings[~ratings.index.isin(train_idx)]

		return trainset, testset

	def _sample_negative(self, stage):
		trainset, testset = self._split(self.ratings)[0], self._split(self.ratings)[1]
		self.interact_hist = {}
		for i, row in trainset.iterrows():
			if row[0] not in self.interact_hist.keys():
				self.interact_hist[row[0]] = [row[1]]
			else:
				self.interact_hist[row[0]].append(row[1])

		if stage == 'train':
			new_trainset = []
			pbar = tqdm(total=trainset.shape[0])
			for i, entry in trainset.iterrows():
				new_trainset.append([entry[0], entry[1], entry[2], random.sample(self.item_pool-set(self.interact_hist[entry[0]]), 20)])
				pbar.update(1)
			pbar.close()
			new_trainset = pd.DataFrame(new_trainset)
			new_trainset.to_csv(os.path.join(self.data_dir,'train.csv'), header=None, index=False)

		elif stage == 'valid':
			new_validset = []
			pbar = tqdm(total=testset.shape[0])
			for i, entry in testset.iterrows():
				new_validset.append([entry[0], entry[1], entry[2], random.sample(self.item_pool-set(self.interact_hist[entry[0]]), 5)])
				pbar.update(1)
			pbar.close()
			validset = pd.DataFrame(new_validset)
			validset.to_csv(os.path.join(self.data_dir,'valid.csv'), header=None, index=False)

		elif stage == 'test':
			pass

	def _build_interact_hist(self):
		trainset = pd.read_csv(os.path.join(self.data_dir,'train.csv'), header=None)
		self.trainset = trainset
		# print(trainset.head())
		self.interact_hist = {}
		for i, row in trainset.iterrows():
			if row[0] not in self.interact_hist.keys():
				self.interact_hist[row[0]] = [row[1]]
			else:
				self.interact_hist[row[0]].append(row[1])

	def instance_a_train_loader(self, num_negatives, batch_size):
		"""instance train loader for one training epoch"""
		users, items, ratings = [], [], []
		train_ratings=self.trainset
		train_ratings.columns=['userId', 'itemId', 'rating', 'negative_samples']

		for row in train_ratings.itertuples():
			users.append(int(row.userId))
			items.append(int(row.itemId))
			ratings.append(float(row.rating))
			negative_samples = row.negative_samples[1:-1].split(',')
			negative_samples = random.sample(negative_samples, num_negatives)
			for i in range(len(negative_samples)):
				users.append(int(row.userId))
				items.append(int(negative_samples[i]))
				ratings.append(float(0))  # negative samples get 0 rating
		dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
										item_tensor=torch.LongTensor(items),
										target_tensor=torch.FloatTensor(ratings))
		return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	# def evaluate_data(self):
	# 	"""create evaluate data"""
	# 	test_ratings = pd.read_csv(self.data_dir+'valid.csv', header=None)
	# 	test_ratings.columns=['userId', 'itemId', 'rating', 'negative_samples']
	# 	test_users, test_items, negative_users, negative_items = [], [], [], []
	# 	for row in test_ratings.itertuples():
	# 		test_users.append(int(row.userId))
	# 		test_items.append(int(row.itemId))
	# 		negative_samples = row.negative_samples[1:-1].split(',')
	# 		for i in range(len(negative_samples)):
	# 			negative_users.append(int(row.userId))
	# 			negative_items.append(int(negative_samples[i]))
	# 	return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
	# 			torch.LongTensor(negative_items)]

	def evaluate_data(self):
		"""create evaluate data"""
		test_ratings = pd.read_csv(os.path.join(self.data_dir,'valid.csv'), header=None)
		test_ratings.columns=['userId', 'itemId', 'rating', 'negative_samples']
		test_users, test_items, negative_users, negative_items = [], [], [], []
		for row in test_ratings.itertuples():
			test_users.append(int(row.userId))
			test_items.append(int(row.itemId))
			# negative_samples = row.negative_samples[1:-1].split(',')
		unique_users = list(test_ratings['userId'].unique())
		print('build negative samples')
		pbar = tqdm(total=len(unique_users))
		for user in unique_users:
			pbar.update(1)
			negative_users.extend([user]*100)
			negative_items.extend(list(random.sample(self.item_pool, 100)))
		pbar.close()
		return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
				torch.LongTensor(negative_items)]

	def instance_a_test_loader(self):
		users, items, ratings=[], [], []
		test_ratings = pd.read_csv(os.path.join(self.data_dir,'valid.csv'), header=None)
		test_ratings.columns=['userId', 'itemId', 'rating', 'negative_samples']
		test_ratings = test_ratings[['userId', 'itemId']]
		users = test_ratings['userId'].tolist()
		dataset = UserDataset(user_tensor=torch.LongTensor(users))
		return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4), self.interact_hist

	def instance_a_rerank_loader(self):
		users, items, ratings=[], [], []
		ranklist_path = os.path.join(self.data_dir,'ranklist_with_gt.json')
		with open(ranklist_path, "r") as f:
			ranklist = json.load(f)
		ranklist_users = ranklist.keys()
		users = []
		items = []
		for idx, user in enumerate(ranklist_users):
			users.extend([int(user)] * len(ranklist[user]))
			for i, item in enumerate(ranklist[user]):
				items.append(int(item))
		dataset = UserItemDataset(
			user_tensor=torch.LongTensor(users),
			item_tensor=torch.LongTensor(items)
			)
		return DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4), self.interact_hist

	def rerank_data(self):
		ranklist_path = os.path.join(self.data_dir,'ranklist_with_gt.json')
		with open(ranklist_path, "r") as f:
			ranklist = json.load(f)
		test_ratings = pd.read_csv(os.path.join(self.data_dir,'valid.csv'), header=None)
		test_ratings.columns=['userId', 'itemId', 'rating', 'negative_samples']
		test_users, test_items, negative_users, negative_items = [], [], [], []
		gt = {}
		for row in test_ratings.itertuples():
			test_users.append(int(row.userId))
			test_items.append(int(row.itemId))
			if row.userId not in gt.keys():
				gt[row.userId] = []
			gt[row.userId].append(row.itemId)
		pbar = tqdm(total = len(ranklist.keys()))
		for key in ranklist.keys():
			ground_truth = gt[int(key)]
			for item in ranklist[key]:
				if item in ground_truth:
					pass
				else:
					negative_users.append(int(key))
					negative_items.append(int(item))
			pbar.update(1)
		pbar.close()
		return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
				torch.LongTensor(negative_items)]



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	args = parser.parse_args()

	df = pd.read_csv(os.path.join(args.input_dir, 'mf.csv'), header=None)
	sample_generator = SampleGenerator(ratings=df, input_dir=args.input_dir)
	sample_generator._sample_negative('train')
	sample_generator._sample_negative('valid')

