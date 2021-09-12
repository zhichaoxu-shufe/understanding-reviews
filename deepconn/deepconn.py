import torch
from torch import LongTensor, FloatTensor
import torch.nn as nn
import torch.nn.functional as F
from layers import NgramFeat, FM, WordEmbedding, LastFeat
from utils import masked_tensor
import sys


class DeepCoNNppRank(nn.Module):
	def __init__(self, user_size, item_size, vocab_size, kernel_sizes, embedding_dim, hidden_dim, latent_dim, doc_len, 
				pretrained_embeddings, dropout, arch="CNN"):
		super(DeepCoNNppRank, self).__init__()

		self.user_size = user_size
		self.item_size = item_size
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		
		self.word_embeddings =  WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings)
		self.ngram = NgramFeat(kernel_sizes, embedding_dim, hidden_dim,
								   doc_len, arch=arch)
		self.user_embed = torch.nn.Embedding(num_embeddings=user_size, embedding_dim=latent_dim)
		self.user_embed.weight.data /= 10
		self.item_embed = torch.nn.Embedding(num_embeddings=item_size, embedding_dim=latent_dim)
		self.item_embed.weight.data /= 10

		self.user_feat = LastFeat(user_size, hidden_dim, latent_dim, padding_idx=0)
		self.item_feat = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=0)
		self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=0,
					item_padding_idx=0)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.mu = 0.3


	def forward(self, u_ids, i_ids, u_revs, i_revs, u_rev_masks, i_rev_masks):
		"""
		Args: 
			u_revs: [num_item, doc_len]
			i_revs: [num_item, doc_len]
			u_rev_masks: [num_item, doc_len]
			i_rev_masks: [num_item, doc_len]
			u_ids: [num_item]
			i_ids: [num_item]
		"""
		num_item = u_revs.shape[0]

		u_revs = self.word_embeddings(u_revs)
		i_revs = self.word_embeddings(i_revs)
		# get features from reviews
		u_rev_feats = self.ngram(u_revs, u_rev_masks).view(num_item, self.hidden_dim)
		i_rev_feats = self.ngram(i_revs, i_rev_masks).view(num_item, self.hidden_dim) 
		u_feats = self.user_feat(u_rev_feats, u_ids)
		i_feats = self.item_feat(i_rev_feats, i_ids)

		preds = self.fm(u_feats, i_feats, u_ids, i_ids)
		return preds