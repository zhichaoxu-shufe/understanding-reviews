import pandas as pd
import numpy as np
import os, sys, time
from copy import deepcopy
import json, pickle, random
from tqdm import tqdm
import argparse

def build_ranklist(input_dir):
	groundtruth = pd.read_csv(os.path.join(input_dir,'valid.csv'), header=None)
	user_dict = {}
	for i, row in groundtruth.iterrows():
		if row[0] not in user_dict.keys():
			user_dict[row[0]] = [row[1]]
		else:
			user_dict[row[0]].append(row[1])
	ranklist = {}
	with open(os.path.join(input_dir,'ranklist.json'), 'r') as f:
		original_ranklist = json.load(f)
	for userid in original_ranklist.keys():
		groundtruth_user = list(user_dict[int(userid)])
		for index, item in enumerate(groundtruth_user):
			groundtruth_user[index] = str(item)

		for i, item in enumerate(groundtruth_user):
			if item in original_ranklist[userid]:
				pass
			else:
				idx = -i
				original_ranklist[userid][idx] = item

	with open(os.path.join(input_dir,'ranklist_with_gt.json'), 'w') as fp:
		json.dump(original_ranklist, fp)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", type=str, required=True)

	args = parser.parse_args()
	build_ranklist(args.input_dir)