import pandas as pd
import numpy as np
import os, sys, time
from copy import deepcopy
import json
from tqdm import tqdm
import argparse

from data import *

def recursive_filter(input_file, recursive_filter, core):
	df = pd.read_json(input_file, lines=True)
	del df['reviewerName']
	del df['helpful']
	del df['summary']
	del df['reviewTime']
	del df['reviewText']
	cols = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
	df = df[cols]

	entries = df.values.tolist()

	if recursive_filter:
		while recursive_filter >= 1:
			user_dict = {}
			item_dict = {}
			for entry in entries:
				if entry[0] not in user_dict.keys():
					user_dict[entry[0]] = 1
				else:
					user_dict[entry[0]] += 1
				if entry[1] not in item_dict.keys():
					item_dict[entry[1]] = 1
				else:
					item_dict[entry[1]] += 1
			
			user_delete_list = []
			item_delete_list = []
			
			for key, value in user_dict.items():
				if value < core:
					user_delete_list.append(key)
			for key, value in item_dict.items():
				if value < core:
					item_delete_list.append(key)
			
			pbar = tqdm(total=len(user_delete_list))
			for key in user_delete_list:
				del user_dict[key]
				pbar.update(1)
			pbar.close()
			pbar = tqdm(total=len(item_delete_list))
			for key in item_delete_list:
				del item_dict[key]
				pbar.update(1)
			pbar.close()

			keep_list = []
			for i,entry in enumerate(entries):
				if entry[0] in user_dict.keys() and entry[1] in item_dict.keys():
					keep_list.append(i)

			new_entries = []
			for i in keep_list:
				new_entries.append(entries[i])
			entries = new_entries

			user_dict = {}
			item_dict = {}
			for entry in entries:
				if entry[0] not in user_dict.keys():
					user_dict[entry[0]] = 1
				else:
					user_dict[entry[0]] += 1
				if entry[1] not in item_dict.keys():
					item_dict[entry[1]] = 1
				else:
					item_dict[entry[1]] += 1

			if min(user_dict.values()) >= core and min(item_dict.values()) >= core:
				recursive_filter = 0

	return entries

def build_dataset(file_path, output_dir, input_type):
	if input_type == 'csv':
		df = pd.read_csv(file_path, header=None)
		df.columns = ['userId', 'itemId', 'rating', 'timestamp']
		user_dict = {}
		item_dict = {}
		unique_user = list(set(df['userId']))
		unique_item = list(set(df['itemId']))
		for i in range(len(unique_user)):
			user_dict[unique_user[i]] = i
		for i in range(len(unique_item)):
			item_dict[unique_item[i]] = i

		new_df = []
		pbar = tqdm(total=df.shape[0])
		for i, row in df.iterrows():
			new_df.append([user_dict[row[0]], item_dict[row[1]], row[2], row[3]])
			pbar.update(1)
		pbar.close()

		new_df = pd.DataFrame(new_df)
		new_df.to_csv(target_path+'mf.csv', header=None, index=False)

		with open(target_path+'users.json', 'w') as fp:
			json.dump(user_dict, fp)
		with open(target_path+'items.json', 'w') as fp:
			json.dump(item_dict, fp)
	if input_type == 'json':
		df = pd.read_json(file_path, lines=True)
		del df['reviewerName']
		del df['helpful']
		del df['summary']
		del df['reviewTime']
		del df['reviewText']
		cols = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
		df = df[cols]
		df.columns = ['userId', 'itemId', 'rating', 'timestamp']
		user_dict = {}
		item_dict = {}
		unique_user = list(set(df['userId']))
		unique_item = list(set(df['itemId']))
		for i in range(len(unique_user)):
			user_dict[unique_user[i]] = i
		for i in range(len(unique_item)):
			item_dict[unique_item[i]] = i

		new_df = []
		pbar = tqdm(total=df.shape[0])
		for i, row in df.iterrows():
			# new_df.append([user_dict[row[0]], item_dict[row[1]], row[2], row[3]])
			new_df.append([user_dict[row[0]], item_dict[row[1]], 1, row[3]])
			pbar.update(1)
		pbar.close()

		new_df = pd.DataFrame(new_df)
		new_df.to_csv(os.path.join(output_dir,'mf.csv'), header=None, index=False)

		with open(os.path.join(output_dir, 'users.json'), 'w') as fp:
			json.dump(user_dict, fp)
		with open(os.path.join(output_dir, 'items.json'), 'w') as fp:
			json.dump(item_dict, fp)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_file', type=str)
	parser.add_argument('--output_dir', type=str)

	args=parser.parse_args()

	build_dataset(args.input_file, args.output_dir, 'json')
	df = pd.read_csv(os.path.join(args.output_dir, 'mf.csv'), header=None)
	sample_generator = SampleGenerator(ratings=df, input_dir=args.output_dir)
	sample_generator._sample_negative('train')
	sample_generator._sample_negative('valid')
