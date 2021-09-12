import pandas as pd
import numpy as np
from gmf import GMFEngine, MFEngine
from data import SampleGenerator
import sys
import copy
import argparse
import utils
from utils import str2bool
import torch
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--input_dir", type=str, required=True)

	parser.add_argument("--optimizer", default="adam", type=str)
	parser.add_argument("--lr", default=1e-2, type=float)
	parser.add_argument("--device_id", default=0, type=int)
	parser.add_argument("--use_cuda", default=True, type=str2bool)

	parser.add_argument("--epochs", default=5, type=int)
	parser.add_argument("--ranklist", default=True, type=str2bool)
	parser.add_argument("--rerank", default=False, type=str2bool)
	parser.add_argument("--evaluate", default=True, type=str2bool)
	parser.add_argument("--evaluate_step", default=5, type=int)

	parser.add_argument("--num_negative", default=4, type=int)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--l2_regularization", default=0, type=int)
	parser.add_argument("--latent_dim", default=32, type=int)

	parser.add_argument("--save_model", type=str2bool, default=True)

	args = parser.parse_args()

	df = pd.read_csv(os.path.join(args.input_dir,'mf.csv'), header=None, names=['userId', 'itemId', 'rating', 'timestamp'],  engine='python')
	print(df.head())
	print('Range of userId is [{}, {}]'.format(df.userId.min(), df.userId.max()))
	print('Range of itemId is [{}, {}]'.format(df.itemId.min(), df.itemId.max()))

	# DataLoader for training
	sample_generator = SampleGenerator(ratings=df, input_dir=args.input_dir)
	sample_generator._build_interact_hist()

	if args.evaluate:
		evaluate_data = sample_generator.evaluate_data()

	config = vars(args)

	config['num_users'] = df.userId.max()+1
	config['num_items'] = df.itemId.max()+1

	## Specify the exact model
	engine = MFEngine(config)

	with open(os.path.join(args.input_dir,'MF_log.txt'), 'w') as f:
		if args.evaluate:
			hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=0)
		for epoch in range(args.epochs):
			print('Epoch {} starts !'.format(epoch))
			train_loader = sample_generator.instance_a_train_loader(args.num_negative, args.batch_size)
			engine.train_an_epoch(train_loader, epoch_id=epoch)
			if args.evaluate:
				if (epoch+1) % args.evaluate_step == 0:
					hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
					line = 'Epoch: '+str(epoch)+' Hit Ratio: '+str(hit_ratio)+' NDCG: '+str(ndcg)+'\n'
					f.write(line)

	if args.save_model:
		torch.save(engine.model, os.path.join(args.input_dir,'model.pt'))

	if args.ranklist:
		test_loader, interact_hist = sample_generator.instance_a_test_loader()
		engine.output_ranklist_large(test_loader, interact_hist, sample_generator.item_pool, args.input_dir, 'rank')

	if args.rerank:
		rerank_loader, inter_hist = sample_generator.instance_a_rerank_loader()
		engine.output_ranklist_large(rerank_loader, inter_hist, sample_generator.item_pool, args.input_dir, 'rerank')

	print(args.dataset)