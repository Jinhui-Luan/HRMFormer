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
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from util import group
from position_embedding import PositionEmbeddingCoordsSine
from helpers import GenericMLP, ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, get_clones
import IPython


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        '''
        x: (bs, npoint, nsample, d)
        return: (bs, d, npoint)
        '''
        b, n, s, d = x.size()                                           
        x1 = x.permute(0, 1, 3, 2)                                          # x1: (bs, npoint, in_channels, nsample)
        x2 = x1.reshape(-1, d, s)                                           # x2: (bs*npint, in_channels, sample)
        batch_size, _, N = x2.size()
        x3 = F.relu(self.bn1(self.conv1(x2)))                               # x3: (bs*npoint, out_channels, sample)
        x4 = F.relu(self.bn2(self.conv2(x3)))                               # x4: (bs*npoint, out_channels, sample)
        
        x5 = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1)              # x5: (bs*npoint, out_channels)
        x6 = x5.reshape(b, n, -1).permute(0, 2, 1)                          # x6: (bs, out_channels, npoint)

        return x6


class EncoderEmbedding(nn.Module):
    def __init__(self, d_i=3, d_h1=128, d_h2=256, d_h3=512, d_model=1024, nsample=8):
        super().__init__()
        self.conv1 = nn.Conv1d(d_i, d_h1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_h1)
        self.conv2 = nn.Conv1d(d_h1, d_h2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(d_h2)
        self.gather_local_0 = Local_op(in_channels=d_h3, out_channels=d_h3)
        self.gather_local_1 = Local_op(in_channels=d_model, out_channels=d_model)
        self.nsample = nsample

    def forward(self, x):
        xyz = x                                                             # xyz: (bs, 87, 3)    
        x = x.permute(0, 2, 1)                                              # (bs, 3, 87)
        x = F.relu(self.bn1(self.conv1(x)))                                 # (bs, 128, 87)
        x = F.relu(self.bn2(self.conv2(x)))                                 # (bs, 256, 87)
        x = x.permute(0, 2, 1)                                              # (bs, 87, 256)

        output = group(nsample=self.nsample, xyz=xyz, points=x)             # (bs, 87, 8, 2*256)
        output = self.gather_local_0(output)                                # (bs, 512, 87)
        output = output.permute(0, 2, 1)                                    # (bs, 87, 512)

        output = group(nsample=self.nsample, xyz=xyz, points=output)        # (bs, 87, 8, 2*512)
        output = self.gather_local_1(output)                                # (bs, 1024, 87)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads=4, d_ffn=128,
                 dropout=0.1, dropout_attn=None,
                 activation="gelu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, d_ffn, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(d_ffn, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.slf_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.slf_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)

        if return_attn_weights:
            return src, attn_weights

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.slf_attn, "dropout"):
            st += f"attn_dr={self.slf_attn.dropout}"
        return st


class TransformerEncoder(nn.Module):

    def __init__(self, enc_layer, enc_n_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(enc_layer, enc_n_layers)
        self.num_layers = enc_n_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

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
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                n_heads = layer.n_heads
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, n_heads, 1, 1)
                mask = mask.view(bsz * n_heads, n, n)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output


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
                # mask must be tiled to num_heads of the transformer
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

    def __init__(self, d_model, n_heads=4, d_ffn=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.slf_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn = self.crs_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if return_attn_weights:
            return tgt, attn

        return tgt, None

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.slf_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2, attn = self.crs_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        if return_attn_weights:
            return tgt, attn

        return tgt, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)


class TransformerDecoder(nn.Module):

    def __init__(self, dec_layer, dec_n_layers, norm_fn_name="ln",
                return_intermediate=False,
                weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(dec_layer, dec_n_layers)
        self.num_layers = dec_n_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1) # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_attn_weights=return_attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, args):

        super().__init__()
        self.enc_embedding = EncoderEmbedding(
            d_i=args.d_i,
            d_h1=args.enc_emb_d_h1,
            d_h2=args.enc_emb_d_h2,
            d_h3=args.enc_emb_d_h3,
            d_model=args.d_model,
            nsample=args.nsample
        )
    
        self.enc_layer = TransformerEncoderLayer(
            d_model=args.enc_d_model,
            n_heads=args.enc_n_heads,
            d_ffn=args.enc_d_ffn,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )

        self.encoder = TransformerEncoder(
            enc_layer=self.enc_layer, 
            enc_n_layers=args.enc_n_layers
        )

        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [args.enc_d_model]
        else:
            hidden_dims = [args.enc_d_model, args.enc_d_model]

        self.enc2dec_prj = GenericMLP(
            input_dim=args.enc_d_model,
            hidden_dims=hidden_dims,
            output_dim=args.dec_d_model,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=args.dec_d_model, 
            pos_type=args.position_embedding, 
            normalize=False
        )

        self.query_prj = GenericMLP(
            input_dim=args.dec_d_model,
            hidden_dims=[args.dec_d_model],
            output_dim=args.dec_d_model,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.num_queries = args.num_queries

        self.dec_layer = TransformerDecoderLayer(
            d_model=args.dec_d_model,
            n_heads=args.dec_n_heads,
            d_ffn=args.dec_d_ffn,
            dropout=args.dec_dropout,
            activation=args.dec_activation
        )

        self.decoder = TransformerDecoder(
            dec_layer=self.dec_layer,
            dec_n_layers=args.dec_n_layers
        )

        self.trg_prj = nn.Linear(args.dec_d_model, args.d_o, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        'To facilitate the residual connections, the dimensions of all module outputs shall be the same.'

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_prj(pos_embed)
        return query_xyz, query_embed


    def forward(self, x, encoder_only=False):
        pre_enc_features = self.enc_embedding(x)                                            # (bs, channel, npoints)

        # nn.MultiHeadAttention in encoder expects features of size (npoints, bs, channel)
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        enc_xyz, enc_features = self.encoder(pre_enc_features, xyz=x)                       # (npoints, bs, channel)

        enc_features = enc_features.permute(1, 2, 0)                                        # (bs, channel, npoint)     
        enc_features = self.enc2dec_prj(enc_features)                                       # (bs, channel, npoint)
        enc_features = enc_features.permute(2, 0, 1)                                        # (npoints, bs, channel) 

        if encoder_only:
            return enc_xyz, enc_features.transpose(0, 1)                                    # (bs, npoints, channel)

        point_cloud_dims = [enc_features.shape[-1], enc_features.shape[-1]] 

        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: (bs, channel, npoints)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # nn.MultiHeadAttention in decoder expects features of size (npoints, bs, channel)
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        trg = torch.zeros_like(query_embed)

        dec_features, _ = self.decoder(trg, enc_features, query_pos=query_embed, pos=enc_pos)   # (j, bs, channel)


        y = self.trg_prj(dec_features.permute(1, 0, 2))

        return y

    
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
