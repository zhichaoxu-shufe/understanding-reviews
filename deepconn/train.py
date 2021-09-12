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
from engine import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", type=str, required=True)
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--log_dir", type=str, default="logs")
	parser.add_argument("--log", type=str2bool, default=False)
	parser.add_argument("--log_idx", type=int, default=500)
	parser.add_argument("--pretrained_path", type=str, required=True)

	parser.add_argument("--use_pretrain", type=str2bool, default=False)
	parser.add_argument("--verbose", type=str2bool, default=False)
	parser.add_argument("--stats", type=str2bool, default=False)
	parser.add_argument("--stats_idx", type=int, default=2)
	parser.add_argument("--parallel", type=str2bool, default=False)
	parser.add_argument("--evaluate", type=str2bool, default=False)
	parser.add_argument("--evaluate_step", type=int, default=1)
	parser.add_argument("--train", type=str2bool, default=True)
	parser.add_argument("--test", type=str2bool, default=True)

	# hyperparameters
	parser.add_argument("--kernel_size", type=int, default=3)
	parser.add_argument("--hidden_dim", type=int, default=10)
	parser.add_argument("--embedding_dim", type=int, default=300)
	parser.add_argument("--latent_dim", type=int, default=10)
	parser.add_argument("--dropout", type=float, default=0.5)
	parser.add_argument("--arch", type=str, default="CNN")

	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--lr_decay", type=float, default=0.0)
	parser.add_argument("--decay_patience", type=int, default=0)
	parser.add_argument("--max_grad_norm", type=float, default=5.0)
	parser.add_argument("--patience", type=int, default=5)

	parser.add_argument("--model_name", type=str, default="DeepCoNNpp")

	args = parser.parse_args()
	config = vars(args)

	if args.train:
		train_dataset = DeepCoNNRankDataset(config, "train")
		train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=2)
		dataloaders = {"train": train_dataloder, "valid": None, "test": None}
		experiment = DeepCoNNExperiment(config, dataloaders)
		experiment.train()

		torch.save(experiment.model, os.path.join(args.input_dir,'deepconn_model.pt'))
		del train_dataset, train_dataloder, experiment

	if args.test:
		test_dataset = DeepCoNNRankDataset(config, "test")
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=8)
		dataloaders = {"train": None, "valid": None, "test": test_dataloader}
		print('finish loading testset')
		experiment = DeepCoNNExperiment(config, dataloaders)
		experiment.model = torch.load(os.path.join(args.input_dir,'deepconn_model.pt'))
		print('finish loading model')
		experiment.test()