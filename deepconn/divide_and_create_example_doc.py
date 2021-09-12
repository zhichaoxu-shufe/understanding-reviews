import json 
import argparse
from collections import defaultdict
import os
import pickle
import gzip
import re
import sys

import pandas as pd
from tqdm import tqdm
import numpy as np

import sys
import copy

from _tokenizer import Vocab, Indexlizer
from _stop_words import ENGLISH_STOP_WORDS


"""
NOTE:
	- Concatenate all reviews from each user or item to form a giant document for models like DeepCoNN, D-Att.
"""
def set_negative_samples(train=4, test=1000):
	train_num=train
	test_num=test
	return train_num, test_num


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string) 
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()



def truncate_pad_tokens(tokens, max_seq_len, pad_token):
	# truncate 
	if len(tokens) > max_seq_len:
		tokens = tokens[:max_seq_len]
	# pad
	res_length = max_seq_len - len(tokens)
	tokens = tokens + [pad_token] * res_length
	return tokens

def write_pickle(path, data):
	with open(path, "wb") as f:
		pickle.dump(data, f)

def split_data(args):
	path = args.data_path
	dest_dir = args.dest_dir

	f = gzip.open(path)

	users = [] 
	items = [] 
	ratings = [] 
	reviews = [] 
	times = []

	for line in f:
		js_dict = json.loads(line)
		if str(js_dict['reviewerID'])=='unknown':
			print("unknown user")
			continue
		if str(js_dict['asin'])=='unknown':
			print("unknown item")
			continue

		users.append(js_dict["reviewerID"])
		items.append(js_dict["asin"])
		ratings.append(js_dict["overall"])
		reviews.append(js_dict["reviewText"])
		times.append(js_dict["unixReviewTime"])

	df = pd.DataFrame({"user_id": pd.Series(users),
						"item_id": pd.Series(items),
						"rating": pd.Series(ratings),
						"review": pd.Series(reviews),
						"time": pd.Series(times)})
	
	# numberize user and item
	user_mapping = {}
	item_mapping = {}
	unique_userid = list(df['user_id'].unique())
	unique_itemid = list(df['item_id'].unique())
	for i, userid in enumerate(unique_userid):
		user_mapping[userid] = i
	for i, itemid in enumerate(unique_itemid):
		item_mapping[itemid] = i
	df_values = df.values.tolist()
	del df
	new_df = []
	for i, entry in enumerate(df_values):
		new_df.append([user_mapping[entry[0]], item_mapping[entry[1]], entry[2], entry[3], entry[4]])
	del df_values
	df = pd.DataFrame(new_df)
	del new_df
	df.columns = ['user_id', 'item_id', 'rating', 'review', 'time']

	# user level ratio split
	df = df.sort_values(['user_id']).reset_index(drop=True)
	def ratio_split(grp, test_size=0.3):
		start_idx = grp.index[0]
		split_len = int(np.ceil(len(grp) * (1 - test_size)))
		split_idx = start_idx + split_len
		end_indx = grp.index[-1]
		if len(grp) <= 2:
			return [start_idx]
		else:
			return list(range(start_idx, split_idx))
	middle = df.groupby('user_id').apply(ratio_split)
	train_idx = df.groupby('user_id').apply(ratio_split).explode().values
	train_df = df.loc[train_idx, :]
	test_df = df[~df.index.isin(train_idx)]
	valid_df = copy.deepcopy(test_df)

	write_pickle(os.path.join(args.dest_dir, "raw_total_df.pkl"), df)
	write_pickle(os.path.join(args.dest_dir, "raw_train_df.pkl"), train_df)
	write_pickle(os.path.join(args.dest_dir, "raw_valid_df.pkl"), valid_df)
	write_pickle(os.path.join(args.dest_dir, "raw_test_df.pkl"), test_df)

	return df, train_df, valid_df, test_df

def create_meta(df, args):
	meta = {}
	# statistics
	reviews = list(df.review)
	indexlizer = Indexlizer(reviews, special_tokens=["<pad>", "<unk>", "<sep>"], preprocessor=clean_str, mode="word",
						stop_words=ENGLISH_STOP_WORDS, max_len=args.max_doc_len)
	#indexlized_reviews = indexlizer.transform(reviews)
	#df["idxed_review"] = indexlized_reviews
	meta["user_num"] = df.user_id.max() + 1
	meta["item_num"] = df.item_id.max() + 1 # 加上 pad_idx 0, 并且考虑了空隙
	print(df.user_id.max(), df.item_id.max())

	# create meta for reviews and rids
	user_docs = {}
	item_docs = {}
	
	# first: form giant document
	train_users = list(df["user_id"])
	train_items = list(df["item_id"])
	train_reviews = list(df["review"])

	for user, item, review in zip(train_users, train_items, train_reviews):
		if user not in user_docs:
			user_docs[user] = review + " <sep> "
		else:
			user_docs[user] += review + " <sep> "
		if item not in item_docs:
			item_docs[item] = review + " <sep> "
		else:
			item_docs[item] += review + " <sep> "
	
	# second: indexlize giant document
	print('indexlize giant document')
	for user, doc in tqdm(user_docs.items()):
		idxed_doc = indexlizer.transform([doc])[0]
		user_docs[user] = idxed_doc
	for item, doc in tqdm(item_docs.items()):
		idxed_doc = indexlizer.transform([doc])[0]
		item_docs[item] = idxed_doc

	item_id_all = [i for i in range(meta['item_num'])]
	for i, item_id in enumerate(item_id_all):
		if item_id not in item_docs.keys():
			item_docs[item_id] = "<pad>" * args.max_doc_len
	
	meta["user_docs"] = user_docs
	meta["item_docs"] = item_docs
	meta["indexlizer"] = indexlizer
	meta["doc_len"] = args.max_doc_len

	# test 
	t_uid, t_iid = 1, 45 
	t_doc = meta["user_docs"][t_uid]
	print("uid: ", t_uid)
	print("idxed decoded doc: ",  t_doc)
	print("decoded doc: ", indexlizer.transform_idxed_review(t_doc))
	print(len(meta["user_docs"][t_uid]))

	t_doc = meta["user_docs"][t_iid]
	print("iid: ", t_iid)
	print("decoded doc: ",  t_doc)
	print("decoded doc: ", indexlizer.transform_idxed_review(t_doc))
	print(len(meta["item_docs"][t_uid]))

	# sys.exit()
	return meta 

def create_examples(df, meta, set_name, args):
	examples = []
	for _, row in tqdm(df.iterrows()):
		uid = row.user_id 
		iid = row.item_id 
		rating = row.rating
		u_doc = meta["user_docs"][uid]
		i_doc = meta["item_docs"][iid]

		exp = [uid, iid, rating, u_doc, i_doc]
		examples.append(exp)
	
	return examples

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", required=True)
	parser.add_argument("--dest_dir", required=True)
	parser.add_argument("--rv_num_keep_prob", default=0.9, type=float)
	parser.add_argument("--max_doc_len", default=500, type=int)
	parser.add_argument("--random_shuffle", default=False)

	args = parser.parse_args()
	if not os.path.exists(args.dest_dir):
		os.makedirs(args.dest_dir)

	df, train_df, valid_df, test_df = split_data(args)
	meta = create_meta(train_df, args)

	train_examples = create_examples(train_df, meta, "train", args)
	valid_examples = create_examples(valid_df, meta, "valid", args)
	test_examples = create_examples(test_df, meta, "test", args)

	# print meta 
	for k, v in meta.items():
		if isinstance(v, dict):
			print(k)
		else:
			print(k, v)

	write_pickle(os.path.join(args.dest_dir, "meta.pkl"), meta)
	write_pickle(os.path.join(args.dest_dir, "train_examples.pkl"), train_examples)
	write_pickle(os.path.join(args.dest_dir, "valid_examples.pkl"), valid_examples)
	write_pickle(os.path.join(args.dest_dir, "test_examples.pkl"), test_examples)