import json
import os, sys
from math import log
from six.moves import xrange
import pandas as pd
import time
import argparse
from utils import *

#compute ndcg
def metrics(doc_list, rel_set):
	dcg = 0.0
	hit_num = 0.0
	for i in xrange(len(doc_list)):
		if doc_list[i] in rel_set:
			#dcg
			dcg += 1/(log(i+2)/log(2))
			hit_num += 1
	#idcg
	idcg = 0.0
	for i in xrange(min(len(rel_set),len(doc_list))):
		idcg += 1/(log(i+2)/log(2))
	ndcg = dcg/idcg
	recall = hit_num / len(rel_set)
	precision = hit_num / len(doc_list)
	#compute hit_ratio
	hit = 1.0 if hit_num > 0 else 0.0
	large_rel = 1.0 if len(rel_set) > len(doc_list) else 0.0
	return recall, ndcg, hit, large_rel, precision

def print_metrics_with_rank_cutoff(rank_cutoff):
	ndcgs = 0.0
	recalls = 0.0
	hits = 0.0
	large_rels = 0.0
	precisions = 0.0
	count_query = 0
	for qid in ranklist.keys():
		if qid in qrel_map.keys():
			recall, ndcg, hit, large_rel, precision = metrics(ranklist[qid][:rank_cutoff], qrel_map[qid])
			count_query += 1
			ndcgs += ndcg
			recalls += recall
			hits += hit
			large_rels += large_rel
			precisions += precision

	print("Query Number:" + str(count_query))
	print("Larger_rel_set@"+str(rank_cutoff) + ":" + str(large_rels/count_query))
	print("Recall@"+str(rank_cutoff) + ":" + str(recalls/count_query))
	print("Precision@"+str(rank_cutoff) + ":" + str(precisions/count_query))
	print("NDCG@"+str(rank_cutoff) + ":" + str(ndcgs/count_query))
	print("Hit@"+str(rank_cutoff) + ":" + str(hits/count_query))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--gt", type=str, required=True)
	parser.add_argument("--ranklist", type=str, required=True)

	args = parser.parse_args()
	args = vars(args)

	with open(args['ranklist'], 'r') as f:
		ranklist = json.load(f)
	testset = pd.read_csv(args['gt'], header=None)

	testset = testset.values.tolist()
	qrel_map = {}
	for entry in testset:
		if entry[0] not in qrel_map.keys():
			qrel_map[entry[0]] = set([entry[1]])
		else:
			qrel_map[entry[0]].add(entry[1])

	reformed = {}
	for key in qrel_map.keys():
		key = int(key)
		reformed[str(key)]=[]
		for value in qrel_map[key]:
			value = int(value)
			reformed[str(key)].append(str(value))

	qrel_map = reformed
	print_metrics_with_rank_cutoff(20)