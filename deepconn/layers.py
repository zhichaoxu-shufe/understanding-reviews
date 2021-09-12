import math

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from utils import masked_tensor

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, padding_idx=0, freeze_embeddings=False):
        super(WordEmbedding, self).__init__()

        self.freeze_embeddings = freeze_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.requires_grad = not self.freeze_embeddings
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(pretrained_embeddings)})
        else:
            print("[Warning] not use pretrained embeddings ...")

    def forward(self, inputs):
        out = self.embedding(inputs)     
        return out

class MyConv1d(nn.Module):
    """
    Support:
        multiple kernel sizes
    """
    def __init__(self, kernel_sizes, in_features, out_features):
        super().__init__()

        if type(kernel_sizes) is str:
            # allow kernel_sizes look like "3,4,5"
            kernel_sizes = [int(x) for x in kernel_sizes.split(",")]

        assert out_features % len(kernel_sizes) == 0
        assert all([kz % 2 == 1 for kz in kernel_sizes])
        

        self.out_features_per_kz = out_features // len(kernel_sizes)
        self.list_of_conv1d = nn.ModuleList([
            nn.Conv1d(in_features, self.out_features_per_kz, kz, padding=(kz-1)//2) for kz in kernel_sizes
        ])
    def forward(self, inputs):
        """
        NOTE: is N x C x L format !!!
        Args:
            inputs: [bz, in_features, seq_len]
        Returns:
            outpus: [bz, out_features, seq_len]
        """
        list_of_outpus = []
        for conv1d_layer in self.list_of_conv1d:
            sub_outputs = conv1d_layer(inputs) #[bz, sub_feat, seq_len]
            list_of_outpus.append(sub_outputs)
        outputs = torch.cat(list_of_outpus, dim=1)

        return outputs

class HierPooling(nn.Module):
    """
    Shout out to masiwei 

    First Avg Pooling with given kernels, then Max Pooling on top of the new generated features
    Support: 
        - 
    TODO:
        - Add Highway Layer after avg_pool, before max_pool ?
    """
    def __init__(self, in_features, out_features, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        if in_features != out_features:
            self.proj_layer = nn.Linear(in_features, out_features)
        else:
            self.proj_layer = None
    
    def forward(self, inputs):
        """
        NOTE: use N x C x L format
        Args:
            inputs: [bz, in_features, seq_len]
        
        Returns:
            outputs: [bz, out_features]
        """
        inputs = F.avg_pool1d(inputs, self.kernel_size, stride=1)
        seq_len = inputs.size(2)
        outputs = F.max_pool1d(inputs, seq_len).squeeze(2)
        assert outputs.dim() == 2

        if self.proj_layer is not None:
            outputs = self.proj_layer(outputs)

        return outputs

class NgramFeat(nn.Module):
    def __init__(self, kernel_sizes, in_features, out_features, seq_len, dropout=0., arch="CNN"):
        super().__init__()

        self.arch = arch
        if arch == "CNN":
            print("use CNN archiecture for Ngram.")
            self.feature_layer = nn.Sequential(MyConv1d(kernel_sizes, in_features, out_features),
                                                nn.ReLU(),
                                                nn.MaxPool1d(seq_len))
        elif arch == "HierPooling":
            print("use HierPooling arch for Ngram.")
            assert len(kernel_sizes) == 1
            self.feature_layer = nn.Sequential(HierPooling(in_features, out_features, kernel_sizes[0]),
                                                nn.ReLU())
        else:
            raise ValueError(f"{arch} is not predefined.")
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None 

    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_features]
            input_masks: [bz, seq_len]
        Returns:
            ouptuts: [bz, out_features]
        """
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2) #[bz, feat, seq_len]
        outputs = self.feature_layer(inputs)
        outputs = outputs.contiguous()
        
        return outputs

class LastFeat(nn.Module):
    def __init__(self, vocab_size, feat_size, latent_dim, padding_idx):
        super(LastFeat, self).__init__()

        self.W = nn.Parameter(torch.Tensor(feat_size, latent_dim))
        self.b = nn.Parameter(torch.Tensor(latent_dim))

        self.ebd = nn.Embedding(vocab_size, latent_dim, padding_idx=padding_idx)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.constant_(self.b, bound)
        nn.init.uniform_(self.ebd.weight, -bound, bound)


    def forward(self, text_feat, my_id):
        """
        Args:
            text_feat: [bz, feat_size]
            my_id: [bz]
        """

        out_feat = text_feat @ self.W + self.b + self.ebd(my_id)

        return out_feat

class FM(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, dropout, user_padding_idx, item_padding_idx):
        super(FM, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.Tensor(latent_dim, 1))
        # self.h_2 = nn.Parameter(torch.tensor(latent_dim, 1))

        self.user_bias = nn.Embedding(user_size, 1, padding_idx=user_padding_idx)
        self.item_bias = nn.Embedding(item_size, 1, padding_idx=item_padding_idx)
        self.g_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.h, -bound, bound)
        # nn.init.uniform_(self.h, -bound, bound)
        nn.init.uniform_(self.user_bias.weight, -bound, bound)
        nn.init.uniform_(self.item_bias.weight, -bound, bound)
        nn.init.constant_(self.g_bias, bound)


    def forward(self, u_feat, i_feat, u_id, i_id):
        """
        Args:
            u_feat: [bz, latent_dim*2]
            i_feat: ...
            u_id: [bz]
            i_id: [bz]

        Returns:
            pred: [bz]
        """
        fm = torch.mul(u_feat, i_feat)
        fm = F.relu(fm)
        fm = self.dropout(fm) #[bz, latent_dim]

        # u_bias = self.user_bias(u_id)
        i_bias = self.item_bias(i_id)

        # pred = fm @ self.h + u_bias + i_bias + self.g_bias
        pred = fm @ self.h + i_bias

        return pred
