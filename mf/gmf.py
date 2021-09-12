import torch
from engine import Engine
from utils import use_cuda
import sys
import torch.nn.functional as F


class GMF(torch.nn.Module):
	def __init__(self, config):
		super(GMF, self).__init__()
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim = config['latent_dim']

		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		# torch.nn.init.normal_(self.embedding_user.weight, std=0.01)
		# torch.nn.init.normal_(self.embedding_item.weight, std=0.01)

		self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
		self.logistic = torch.nn.Sigmoid()

	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices)
		item_embedding = self.embedding_item(item_indices)
		element_product = torch.mul(user_embedding, item_embedding)
		logits = self.affine_output(element_product)
		rating = self.logistic(logits)
		return rating

	def init_weight(self):
		pass

class MF(torch.nn.Module):
	def __init__(self, config):
		super(MF, self).__init__()
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim = config['latent_dim']
		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.bias_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)

	def forward(self, user_indices, item_indices):
		user_embedding = self.embedding_user(user_indices) # [bz, latent_dim]
		item_embedding = self.embedding_item(item_indices) # [bz, latent_dim]
		element_product = torch.mul(user_embedding, item_embedding) # [bz, latent_dim]
		element_product = element_product.sum(axis=1)
		item_bias = self.bias_item(item_indices).squeeze()
		# print(element_product.shape, user_bias.shape, item_bias.shape)
		# sys.exit()
		return element_product+item_bias
		

	def init_weight(self):
		# torch.nn.init.normal_(self.embedding_user.weight, std=0.01)
		# torch.nn.init.normal_(self.embedding_item.weight, std=0.01)
		pass

class FM(torch.nn.Module):
	def __init__(self, config):
		super(FM, self).__init__()
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim = config['latent_dim']
		self.h = nn.Parameter(torch.Tensor(self.latent_dim, 1))
		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.bias_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=1)

	def forward(self, user_indices, item_indices):
		u_feat = self.embedding_user(user_indices)
		i_feat = self.embedding_item(item_indices)
		fm = torch.mul(u_feat, i_feat)
		fm = F.relu(fm)
		fm = nn.Dropout(fm)
		i_bias = self.bias_item(item_indices)
		return fm @ self.h + i_bias

class GMFEngine(Engine):
	"""Engine for training & evaluating GMF model"""
	def __init__(self, config):
		self.model = GMF(config)
		if config['use_cuda'] is True:
			use_cuda(True, config['device_id'])
			self.model.cuda()
		super(GMFEngine, self).__init__(config)

class MFEngine(Engine):
	def __init__(self, config):
		self.model = MF(config)
		if config['use_cuda'] is True:
			use_cuda(True, config['device_id'])
			self.model.cuda()
		super(MFEngine, self).__init__(config)