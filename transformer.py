# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from re import X
from turtle import forward
from typing import Optional
import pickle
from matplotlib.pyplot import xlabel
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from util import group
from position_embedding import PositionEmbeddingCoordsSine
from helpers import GenericMLP, ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, get_clones
import IPython


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, d_i=3, d_h1=64, d_h2=256, d_o=1024):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(d_i, d_h1, kernel_size=1),
            nn.BatchNorm1d(d_h1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_h1, d_h2, kernel_size=1),
            nn.BatchNorm1d(d_h2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_h2, d_o, kernel_size=1)
            )

    def forward(self, xyz):
        if xyz is None:
            position_embedding = None
        else:
            xyz = xyz.transpose(1, 2).contiguous()
            position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class Local_op(nn.Module):
    def __init__(self, d_i, d_o):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(d_i, d_o, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_o)
        self.conv2 = nn.Conv1d(d_o, d_o, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(d_o)

    def forward(self, x):
        '''
        x: (B, N, n_sample, d_i)
        return: (B, d_o, N)
        '''
        b, n, s, d = x.size()                                           
        x1 = x.permute(0, 1, 3, 2)                                          # x1: (B, N, d_i, n_sample)
        x2 = x1.reshape(-1, d, s)                                           # x2: (B*N, d_i, n_sample)
        BN, _, N = x2.size()
        x3 = F.relu(self.bn1(self.conv1(x2)))                               # x3: (B*N, d_o, n_sample)
        x4 = F.relu(self.bn2(self.conv2(x3)))                               # x4: (B*N, d_o, n_sample)
        
        x5 = F.adaptive_max_pool1d(x4, 1).view(BN, -1)                      # x5: (B*N, d_o)
        x6 = x5.reshape(b, n, -1).permute(0, 2, 1)                          # x6: (B, d_o, N)

        return x6


class InputEmbedding(nn.Module):
    def __init__(self, d_i=3, d_h1=64, d_h2=256, d_h3=512, d_o=1024, n_sample=8):
        super().__init__()
        self.conv1 = nn.Conv1d(d_i, d_h1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_h1)
        self.conv2 = nn.Conv1d(d_h1, d_h2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(d_h2)
        self.gather_local_0 = Local_op(d_h3, d_h3)
        self.gather_local_1 = Local_op(d_o, d_o)
        self.n_sample = n_sample

    def forward(self, xyz):
        output = xyz                                                        # (B, N, d_i)    
        output = output.permute(0, 2, 1)                                    # (B, d_i, N)
        output = F.relu(self.bn1(self.conv1(output)))                       # (B, d_h1, N)
        output = F.relu(self.bn2(self.conv2(output)))                       # (B, d_h2, N)
        output = output.permute(0, 2, 1)                                    # (B, N, d_h2)

        output = group(n_sample=self.n_sample, xyz=xyz, feature=output)       # (B, N, n_sample, 2*d_h2)
        output = self.gather_local_0(output)                                # (B, d_h3, N)
        output = output.permute(0, 2, 1)                                    # (B, N, d_h3)

        output = group(n_sample=self.n_sample, xyz=xyz, feature=output)       # (B, N, n_sample, 2*d_h3)
        output = self.gather_local_1(output)                                # (B, d_o, N)

        return output


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, activation='gelu', pre_norm=True, norm_name='ln', dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hid) 
        self.linear2 = nn.Linear(d_hid, d_in) 
        self.norm = NORM_DICT[norm_name](d_in)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.activation = ACTIVATION_DICT[activation]()

    def forward(self, x):   # (B, N, C)
        residual = x

        if self.pre_norm:
            x = self.norm(x)

        output = self.dropout(self.activation(self.linear1(x)))
        output = self.dropout(self.linear2(output))
        output += residual

        if not self.pre_norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=1024, d_ffn=2048, n_heads=8, dropout=0.1, dropout_attn=None, 
        activation="gelu", pre_norm=True, norm_name="ln"):
        super().__init__()

        if dropout_attn is None:
            dropout_attn = dropout

        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_attn)

        self.pos_ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_ffn, activation=activation,
            pre_norm=pre_norm, norm_name=norm_name, dropout=dropout)

        self.norm = NORM_DICT[norm_name](d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.activation = ACTIVATION_DICT[activation]()

        self.pre_norm = pre_norm
        self.n_heads = n_heads

    def with_pos_embed(self, x, pos: Optional[Tensor]):
        return x if pos is None else x + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False
        ):

        residual = src

        if self.pre_norm:
            src = self.norm(src)
        
        q = k = self.with_pos_embed(src, src_pos)
        v = src

        output, slf_attn_weights = self.slf_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        output = self.dropout(output)
        output += residual

        if not self.pre_norm:
            output = self.norm(output)

        output = self.pos_ffn(output)

        if return_attn_weights:
            return output, slf_attn_weights
        else:
            return output, None
        
    def extra_repr(self):
        st = ""
        if hasattr(self.slf_attn, "dropout"):
            st += f"attn_dr={self.slf_attn.dropout}"
        return st


class TransformerEncoder(nn.Module):

    def __init__(self, enc_layer, enc_n_layers, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(enc_layer, enc_n_layers)
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                return_attn_weights = False
        ):

        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if src_pos is not None:
                src_pos = src_pos.flatten(2).permute(2, 0, 1)

        output = src

        slf_attns = []

        orig_mask = src_mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [src_mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                n_heads = layer.n_heads
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, n_heads, 1, 1)
                mask = mask.view(bsz * n_heads, n, n)
            output, slf_attn_weights = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, src_pos=src_pos)
            if return_attn_weights:
                slf_attns.append(slf_attn_weights)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        if return_attn_weights:
            return output, torch.stack(slf_attns)
        else:
            return output, None


class MaskedTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, masking_radius, interim_downsampling,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):

        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                # mask must be tiled to n_heads of the transformer
                bsz, n, n = mask.shape
                n_heads = layer.n_heads
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, n_heads, 1, 1)
                mask = mask.view(bsz * n_heads, n, n)

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                # output is npoints x batch x channel. make batch x channel x npoints
                output = output.permute(1, 2, 0)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                # swap back
                output = output.permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds
    
    def extra_repr(self):
        radius_str = ", ".join(["%.2f"%(x) for x in self.masking_radius])
        return f"masking_radius={radius_str}"
        

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=1024, d_ffn=2048, n_heads=8, dropout=0.1, dropout_attn=None, 
        activation="gelu", pre_norm=True, norm_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout

        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_ffn, activation=activation,
                pre_norm=pre_norm, norm_name=norm_name, dropout=dropout)

        self.norm = NORM_DICT[norm_name](d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.activation = ACTIVATION_DICT[activation]()

        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, trg, memory,
                trg_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                trg_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                trg_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False
        ):

        residual = trg

        if self.pre_norm:
            trg = self.norm(trg)

        q = k = self.with_pos_embed(trg, trg_pos)
        v = trg

        output, slf_attn_weights = self.slf_attn(q, k, v, attn_mask=trg_mask, key_padding_mask=trg_key_padding_mask)
        output = self.dropout(output)
        output += residual

        if not self.pre_norm:
            output = self.norm(output)

        residual = output

        if self.pre_norm:
            output = self.norm(output)
        
        q = self.with_pos_embed(output, trg_pos)
        k = self.with_pos_embed(memory, src_pos)
        v = memory

        output, crs_attn_weights = self.crs_attn(q, k, v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        output = self.dropout(output)
        output += residual

        if not self.pre_norm:
            output = self.norm(output)

        output = self.pos_ffn(output)

        if return_attn_weights:
            return output, slf_attn_weights, crs_attn_weights
        else:
            return output, None, None


class TransformerDecoder(nn.Module):

    def __init__(self, dec_layer, dec_n_layers, weight_init_name="xavier_uniform", ):
        super().__init__()
        self.layers = get_clones(dec_layer, dec_n_layers)
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, trg, memory,
                trg_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                trg_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                trg_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                return_intermediate: Optional [bool] = False
        ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1) # memory: bs, c, t -> t, b, c
            if src_pos is not None:
                src_pos = src_pos.flatten(2).permute(2, 0, 1)

        output = trg

        intermediate = []
        slf_attns = []
        crs_attns = []

        for layer in self.layers:
            output, slf_attn_weights, crs_attn_weights = layer(output, memory, trg_mask=trg_mask, memory_mask=memory_mask,
                           trg_key_padding_mask=trg_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           src_pos=src_pos, trg_pos=trg_pos, return_attn_weights=return_attn_weights)
            if return_intermediate:
                intermediate.append(output)
            if return_attn_weights:
                slf_attns.append(slf_attn_weights)
                crs_attns.append(crs_attn_weights)

        if return_attn_weights:
            if return_intermediate:
                return torch.stack(intermediate), torch.stack(slf_attns), torch.stack(crs_attns)
            else:
                return output, torch.stack(slf_attns), torch.stack(crs_attns)
        else:
            if return_intermediate:
                return torch.stack(intermediate), None, None
            else:
                return output, None, None


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, args):

        super().__init__()
        self.enc_embedding = InputEmbedding(
            d_i=args.d_i,
            d_h1=args.d_h1,
            d_h2=args.d_h2,
            d_h3=args.d_h3,
            d_o=args.d_model,
            n_sample=args.n_sample
        )

        self.enc_pos_embedding = PositionEmbeddingLearned(
            d_i=args.d_i,
            d_h1=args.d_h1,
            d_h2=args.d_h2, 
            d_o=args.d_model
        )
    
        self.enc_layer = TransformerEncoderLayer(
            d_model=args.d_model,
            d_ffn=args.d_ffn,
            n_heads=args.n_heads,
            dropout=args.dropout,
            activation=args.activation,
            pre_norm=args.pre_norm,
            norm_name=args.norm_name
        )

        self.encoder = TransformerEncoder(
            enc_layer=self.enc_layer, 
            enc_n_layers=args.enc_n_layers
        )

        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [args.d_model]
        else:
            hidden_dims = [args.d_model, args.d_model]

        self.enc2dec_prj = GenericMLP(
            input_dim=args.d_model,
            hidden_dims=hidden_dims,
            output_dim=args.d_model,
            norm_name=args.norm_name,
            activation=args.activation,
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        self.query_embed = nn.Embedding(args.n_q, args.d_model)

        # self.query_prj = GenericMLP(
        #     input_dim=args.d_model,
        #     hidden_dims=[args.d_model],
        #     output_dim=args.d_model,
        #     norm_name=args.norm_name,
        #     activation=args.activation,
        #     use_conv=True,
        #     output_use_activation=True,
        #     hidden_use_bias=True,
        # )

        # self.dec_pos_embedding = PositionEmbeddingLearned(
        #     d_i=args.d_i,
        #     d_h1=args.d_h1,
        #     d_h2=args.d_h2, 
        #     d_o=args.d_model
        # )

        # self.n_q = args.n_q

        self.dec_layer = TransformerDecoderLayer(
            d_model=args.d_model,
            d_ffn=args.d_ffn,
            n_heads=args.n_heads,
            dropout=args.dropout,
            activation=args.activation,
            pre_norm=args.pre_norm,
            norm_name=args.norm_name
        )

        self.decoder = TransformerDecoder(
            dec_layer=self.dec_layer,
            dec_n_layers=args.dec_n_layers
        )

        self.trg_prj = nn.Linear(args.d_model, args.d_o, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        'To facilitate the residual connections, the dimensions of all module outputs shall be the same.'

    # def get_query_embeddings(self, xyz):
    #     query_inds = furthest_point_sample(xyz, self.n_q)
    #     query_inds = query_inds.long()
    #     query_xyz = [torch.gather(xyz[..., x], 1, query_inds) for x in range(3)]
    #     query_xyz = torch.stack(query_xyz)
    #     query_xyz = query_xyz.permute(1, 2, 0)

    #     # Gater op above can be replaced by the three lines below from the pointnet2 codebase
    #     # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
    #     # query_xyz = gather_operation(xyz_flipped, query_inds.int())
    #     # query_xyz = query_xyz.transpose(1, 2)
    #     query_pos = self.dec_pos_embedding(query_xyz)
    #     query_embed = self.query_prj(query_pos)
    #     return query_embed.permute(2, 0, 1), query_pos.permute(2, 0, 1)


    def forward(self, xyz, encoder_only=False):
        src_pos = self.enc_pos_embedding(xyz)                                               # (B, C, N)
        src_pos = src_pos.permute(2, 0, 1)

        pre_enc_features = self.enc_embedding(xyz)                                          # (B, C, N)
        # nn.MultiHeadAttention in encoder expects features of size (N, B, C)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        enc_features = self.encoder(pre_enc_features, src_pos=src_pos)[0]                   # (N, B, C)
        enc_features = enc_features.permute(1, 2, 0)                                        # (B, C, N)     
        enc_features = self.enc2dec_prj(enc_features)                                       # (B, C, N)
        enc_features = enc_features.permute(2, 0, 1)                                        # (N, B, C) 

        if encoder_only:
            return enc_features.permute(1, 0, 2)                                            # (B, N, C)

        trg_pos = self.query_embed.weight                                                   # (n_q, C)
        trg_pos = trg_pos.unsqueeze(1).repeat(1, xyz.shape[0], 1)                           # (n_q, B, C)                                
        trg = torch.zeros_like(trg_pos)
        
        # trg, trg_pos = self.get_query_embeddings(xyz)                                       # (n_q, B, C)

        # nn.MultiHeadAttention in decoder expects features of size (N, B, C)
        dec_features = self.decoder(trg, enc_features, src_pos=src_pos, trg_pos=trg_pos)[0]         # (n_q, B, C)

        output = self.trg_prj(dec_features.permute(1, 0, 2))                                        # (B, n_q, C)

        return output

    
class SMPLModel_torch(nn.Module):
    def __init__(self, model_path, device=None):
        super(SMPLModel_torch, self).__init__()
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].todense())).type(torch.float64)
        if 'joint_regressor' in params.keys():
            self.joint_regressor = torch.from_numpy(np.array(params['joint_regressor'].T.todense())).type(torch.float64)
        else:
            self.joint_regressor = torch.from_numpy(np.array(params['J_regressor'].todense())).type(torch.float64)
        self.weights = torch.from_numpy(params['weights']).type(torch.float64)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64)
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float64)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float64)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        self.verts = None
        self.joints = None
        for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            # print(' Tensor {} shape: '.format(name), _tensor.shape)
            setattr(self, name, _tensor.to(device))

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.
        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].
        """
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
        m = torch.stack((z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        Parameter:
        ---------
        x: Tensor to be appended.
        Return:
        ------
        Tensor after appending of shape [4,4]
        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
        ).expand(x.shape[0],-1,-1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]
        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.
        """
        zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float64).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans=None, simplify=False):
        """
        Construct a compute graph that takes in parameters and outputs a tensor as
        model vertices. Face indices are also returned as a numpy ndarray.
        
        20190128: Add batch support.
        Parameters:
        ---------
        pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.
        betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
        PCA components. Only 10 components were released by SMPL author.
        trans: Global translation tensor of shape [N, 3].
        Return:
        ------
        A 3-D tensor of [N * 6890 * 3] for vertices
        """
        batch_num = betas.shape[0]
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(betas.to(torch.float64), self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
                torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float64)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        g = []
        g.append(self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            g.append(
                torch.matmul(
                    g[parent[i]],
                    self.with_zeros(
                        torch.cat((R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))), dim=2)
                    )
                )
            )
        

        g = torch.stack(g, dim=1)

        G = g - \
            self.pack(
                torch.matmul(
                    g,
                    torch.reshape(
                        torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float64).to(self.device)), dim=2),
                        (batch_num, 24, 4, 1)
                    )     
                )
            )
        # Restart from here
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=torch.float64).to(self.device)), dim=2
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]

        if trans == None:
            self.verts = v
            self.joints = g[:, :, :3, 3]
        else:
            self.verts = v + torch.reshape(trans, (batch_num, 1, 3))
            self.joints = g[:, :, :3, 3] + torch.reshape(trans, (batch_num, 1, 3))
