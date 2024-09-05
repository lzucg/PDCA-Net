import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
import numbers
from torchvision.ops import DeformConv2d
# -------------------------------------------------------------------#

from .init_weights import init_weights

# https://github.com/XLearning-SCU/2023-CVPR-CODE

class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_




class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim, squeezes, shuffle, expan_att_chans):
        super(CondensedAttentionNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        self.ch_sp_squeeze = nn.Sequential(
            nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
        )

        self.channel_attention = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.spatial_attention = SpatialAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)

        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x):
        x = self.ch_sp_squeeze(x)

        group_num = self.sque_ch_dim
        each_group = self.sque_sp_dim // self.sque_ch_dim
        idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        x = x[:, idx, :, :]

        x = self.channel_attention(x)
        nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = x[:, nidx, :, :]

        x = self.spatial_attention(x)
        x = self.sp_ch_unsqueeze(x)
        return x


class DualAdaptiveNeuralBlock(nn.Module):
    def __init__(self, embed_dim):
        super(DualAdaptiveNeuralBlock, self).__init__()
        self.embed_dim = embed_dim

        self.group_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Conv2d(embed_dim, embed_dim * 2, 7, 1, 3, groups=embed_dim)
        )
        self.post_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x0, x1 = self.group_conv(x).view(B, C, 2, H, W).chunk(2, dim=2)
        x_ = F.gelu(x0.squeeze(2)) * torch.sigmoid(x1.squeeze(2))
        x_ = self.post_conv(x_)
        return x_


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops



class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):

        b, c, w, h = x.shape
        x = x.view(b, w*h, c)
        # bs x hw x c

        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)
        b, wh, c = x.shape

        x = rearrange(x, 'b (h w) c -> b c w h', h=int(math.sqrt(wh)), w=int(math.sqrt(wh)))

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)


class ChannelAttentionBlock(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttentionBlock(num_feat, reduction),
        )

    def forward(self, x, x_size):
        x = self.cab(x.contiguous())
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 squeezes,shuffle,
                 expan_att_chans,mlp_ratio=4.,drop=0.,act_layer=nn.GELU
                 ):
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ca = CondensedAttentionNeuralBlock(embed_dim, squeezes, shuffle, expan_att_chans)

        self.conv = CAB(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.da = DualAdaptiveNeuralBlock(embed_dim)

        self.mlp = LeFF(embed_dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.size()
        x_ = self.norm1(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W).contiguous()

        x_cab = self.conv(x_ , (H, W))
        x = x + self.ca(x_)

        x = x + x_cab
        x_ = self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W).contiguous()
        x = x + self.da(x_)
        return x

# Muti_scale_Feature_Restore 多尺度特征恢复
class MSFR(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(MSFR, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


groups = 32

class Conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Cross_Att(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.transformer_s = TransformerBlock(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_l = TransformerBlock(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_s = nn.Linear(dim_s, dim_l)
        self.linear_l = nn.Linear(dim_l, dim_s)

    def forward(self, e, r):
       b_e, c_e, h_e, w_e = e.shape
       e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)
       b_r, c_r, h_r, w_r = r.shape
       r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)
       e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1,2)), 1)
       r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1,2)), 1)
       e_t = self.linear_l(e_t).unsqueeze(1)
       r_t = self.linear_s(r_t).unsqueeze(1)
       r = self.transformer_s(torch.cat([e_t, r],dim=1))[:, 1:, :]
       e = self.transformer_l(torch.cat([r_t, e],dim=1))[:, 1:, :]
       e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e)
       r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r)
       return e, r


class CODEFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, embed_dim=64, expan_att_chans=4,
                 refine_blocks=4, num_blocks=(6, 4, 4, 2), num_shuffles=(16, 8, 4, 2),
                 ch_sp_squeeze=[(4, 8), (4, 4), (4, 2), (4, 1)]):
        super(CODEFormer, self).__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        filters = [64, 128, 256, 512]
        self.encoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans
            )
            for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))])

        self.is_batchnorm =True

        self.decoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans
            ) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))][::-1])

        self.downsampler = nn.ModuleList([nn.Sequential(
            nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1)), 3, 1, 1),
            nn.PixelUnshuffle(2)
        ) for i in range(len(num_blocks) - 1)]).append(nn.Identity())

        self.upsampler = nn.ModuleList([nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(embed_dim * 2 ** (i - 1)), int(embed_dim * 2 ** i), 3, 1, 1)
        ) for i in range(len(num_blocks) - 1)][::-1]).append(nn.Identity())

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                embed_dim, ch_sp_squeeze[0], num_shuffles[0], expan_att_chans
            ) for _ in range(refine_blocks)
        ])

        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

    def forward(self, x):
        x_emb = self.patch_embed(x)

        # Encoder
        x_ = x_emb
        x_ms = []
        for layer, sampler in zip(
                self.encoder, self.downsampler
        ):

            x_temp_enc = x_
            x_ = layer(x_)

            x_ = x_ + x_temp_enc
            # [1, 64, 224, 224]
            # 加入注意力
            B, C, H, W = x_.shape


            x_ms.append(x_)
            x_ = sampler(x_)


        # Decoder
        x_ = 0
        x_ms.reverse()
        index = 0
        for x_e, layer, sampler in zip(
                x_ms, self.decoder, self.upsampler
        ):

            x_temp_dec = x_e

            x_t = layer(x_ + x_e)
            x_t = x_t + x_temp_dec
            # 加入注意力
            B, C, H, W = x_t.shape

            x_ = sampler(x_t)

            index += 1
        x_ = x_ + x_emb

        B, C, H, W = x_.shape


        # Refinement
        x_ = self.refinement(x_)
        x_ = self.conv_last(x_)
        return x_


#
# ##########################################################################
# ## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(pl.LightningModule):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(pl.LightningModule):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ##########################################################################
# ## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(pl.LightningModule):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

#
# ##########################################################################
# ## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = stride
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=self.stride, padding=1, groups=dim * 2,
                                   bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(x))

        b, f, h1, w1 = q.size()

        q = rearrange(q, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



if __name__ == '__main__':
    model = CODEFormer(out_chans=1)
    from ptflops import get_model_complexity_info

    #model = dehazeformer_l()
    #macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((1, 3, 224, 224))
    out = model(x).to(device)
    print(out.shape)
