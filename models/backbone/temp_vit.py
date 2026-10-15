# Adapted from ViTPose / mmpose (Apache-2.0)
# https://github.com/ViTAE-Transformer/ViTPose
import math
from typing import Optional
import torch
from functools import partial
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from utils.grid_interpolate import coords2heatmap

from .basic_vit import ConvEmbed, Block, Mlp, DropPath, CrossAtten, trunc_normal_

# import sys
# sys.path.insert(0, '.')
from config import cfg

# triplane heatmaps are normalised against a fixed reference batch so that the
# plane features do not scale with the batch size used at inference
REF_BATCH = 64

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size = 3,
        padding = 1, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, stride, padding)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class SeqIdentity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return x

class CrossAttnOffset(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None,
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads


        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, all_head_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.pose_proj_1 = nn.Linear(dim, 3)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def attn(self, x, content):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        B, T, C = content.shape
        kv = self.kv(content).reshape(B, T, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn_ = self.attn_drop(attn)

        x = (attn_ @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        pos_1 = self.pose_proj_1(x)

        return x, pos_1

    def forward(self, x, content):
        x, pos_1 = self.attn(self.norm1(x), self.norm1(content))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, pos_1


class SpatioTemporalModule(nn.Module):
    def __init__(self, dim, num_heads, drop_path_rate=0., drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        self.spatio_block = Block(dim, num_heads, drop=drop, attn_drop=drop, drop_path=drop_path_rate)

        self.temporal_block = Block(dim, num_heads, drop=drop, attn_drop=drop, drop_path=drop_path_rate)

    def forward(self, x):
        B, T, J, C = x.shape

        x = self.spatio_block(x.view(B*T, J, C)).view(B, T, J, C)
        x = x.permute(0,2,1,3).contiguous()
        x = self.temporal_block(x.view(B*J, T, C)).view(B, T, J, C)

        return x


class Triplane(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., drop_path_rate=0., drop=0., cls_n = 6, heatmap_size=16) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.cls_n = cls_n

        self.heatmap_size = heatmap_size
        stride = (heatmap_size // 8)
        kernel_size = 3 if stride != 4 else 7
        padding = 1 if stride !=4 else 3

        self.scale = qk_scale or head_dim ** -0.5

        self.seq_cls_attn = CrossAttnOffset(dim, num_heads, 4, qkv_bias, qk_scale, proj_drop,
            attn_drop)

        self.depth_attn = CrossAtten(dim, num_heads, 4, qkv_bias, qk_scale, proj_drop,
            attn_drop)

        heatmap_encoding = positionalencoding2d(dim, heatmap_size, heatmap_size)
        self.register_buffer("heatmap_encoding", heatmap_encoding)

        self.xy_conv = BasicBlock(dim, dim, kernel_size, padding, stride, downsample=nn.AvgPool2d(stride))
        self.yz_conv = BasicBlock(dim, dim, kernel_size, padding, stride, downsample=nn.AvgPool2d(stride))
        self.xz_conv = BasicBlock(dim, dim, kernel_size, padding, stride, downsample=nn.AvgPool2d(stride))
        
        self.offset_proj = nn.Linear(dim, 3)
        self.offset_proj_finger = nn.Linear(dim, 12)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.seq_norm = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, dim*4, dim, nn.GELU, drop)

        self.pos_hook = nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def interpolate(self, cls, kp_uvd, feat_size):
        B, J, C = cls.shape
        heat_size = self.heatmap_size

        xy_heatmap = coords2heatmap(kp_uvd[:,:,:2], heat_size, heat_size)
        xy_heatmap = xy_heatmap / (xy_heatmap.sum(dim=(1,2), keepdim=True) * REF_BATCH)
        L = xy_heatmap.shape[-1]
        cls_ = cls.unsqueeze(2).repeat(1,1,L,1)
        xy_feat = torch.sum(cls_ * xy_heatmap.unsqueeze(-1), dim=1).permute(0,2,1).reshape(B, -1, heat_size, heat_size)
        xy_feat += self.heatmap_encoding.repeat(B, 1, 1, 1)
        xy_feat = self.xy_conv(xy_feat)
        xy_feat = xy_feat.view(B, self.dim, -1).permute(0,2,1)

        xz_heatmap = coords2heatmap(kp_uvd[:,:,[0,2]], heat_size, heat_size)
        xz_heatmap = xz_heatmap / (xz_heatmap.sum(dim=(1,2), keepdim=True) * REF_BATCH)
        xz_feat = torch.sum(cls_* xz_heatmap.unsqueeze(-1), dim=1).permute(0,2,1).reshape(B, -1, heat_size, heat_size)
        xz_feat += self.heatmap_encoding.repeat(B, 1, 1, 1)
        xz_feat = self.xz_conv(xz_feat)
        xz_feat = torch.cat([torch.mean(xz_feat, dim=2), torch.mean(xz_feat, dim=3)], dim=2)
        xz_feat = xz_feat.permute(0, 2, 1)

        yz_heatmap = coords2heatmap(kp_uvd[:,:,1:], heat_size, heat_size)
        yz_heatmap = yz_heatmap / (yz_heatmap.sum(dim=(1,2), keepdim=True) * REF_BATCH)
        yz_feat = torch.sum(cls_* yz_heatmap.unsqueeze(-1), dim=1).permute(0,2,1).reshape(B, -1, heat_size, heat_size)
        yz_feat += self.heatmap_encoding.repeat(B, 1, 1, 1)
        yz_feat = self.yz_conv(yz_feat)
        yz_feat = torch.cat([torch.mean(yz_feat, dim=2), torch.mean(yz_feat, dim=3)], dim=2)
        yz_feat = yz_feat.permute(0, 2, 1) 

        return xy_feat, xz_feat, yz_feat

    def forward(self, x, seq):
        cls, x = x[:,:self.cls_n], x[:,self.cls_n:]
        B, T, J, C = seq.shape

        seq_ = seq.permute(0,2,1,3).reshape(-1,T,C)
        cls_ = torch.cat([cls[:,:1], cls[:,1:].repeat_interleave(4,dim=1)], dim=1).reshape(-1, 1, C)
        cls_, jt_uvd = self.seq_cls_attn(cls_, seq_)

        pos = torch.cat([self.offset_proj(cls[:, 0]).unsqueeze(1), \
                    self.offset_proj_finger(cls[:,1:]).view(B, -1, 3)], dim=1) 

        jt_uvd = self.pos_hook(pos + jt_uvd.view(B, -1, 3))
        cls_ = cls_.view(B, -1, C)

        xy_feat, xz_feat, yz_feat = self.interpolate(cls_, jt_uvd, int(math.sqrt(x.shape[1])))

        x = x + self.drop_path(xy_feat)

        x = x + self.drop_path(self.depth_attn(x, torch.cat([xz_feat, yz_feat], dim=1)))
       
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return torch.cat([cls, x], dim=1)
  

class MultiClsSeqViTv2(nn.Module):
    def __init__(self, 
                 img_size=64, patch_size=7, in_chans=3, num_classes=80, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, 
                 drop_path_rate=0, hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True, patch_padding='pad', freeze_attn=False, 
                 freeze_ffn=False, cls_n=6):
        super(MultiClsSeqViTv2, self).__init__()
        # Protect mutable default arguments
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.cls_n = cls_n

        self.cls_token = nn.Parameter(torch.zeros(cls_n, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        self.conv_embed = nn.Sequential(ConvEmbed(7, in_chans, 64, 4, 2, norm_layer),
                                        nn.GELU(),
                                        ConvEmbed(3, 64, embed_dim, 2, 1, norm_layer))
        
        new_dpr = [x.item() for x in torch.linspace(drop_path_rate, 0., depth)]
        seqembed_times = cfg.seqembed_times 
        s_step = depth // seqembed_times
        self.seqblock_index = [depth - s_step*i - 1 for i in range(seqembed_times)][::-1]
        self.seqblocks = nn.ModuleList([
            Triplane(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                        attn_drop=attn_drop_rate, proj_drop=drop_rate, drop_path_rate=new_dpr[i], drop=drop_rate, cls_n=cls_n) \
                        if i in self.seqblock_index else SeqIdentity() for i in range(depth)
        ])
        
        self.root_proj = nn.Linear(embed_dim, 2)
        self.joint_proj = nn.Linear(embed_dim, 8)
        
        self.init_weights()
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def forward(self, x, t_token=None):
        B, C, H, W = x.shape
        x = self.conv_embed(x).view(B,self.embed_dim,-1).transpose(2,1).contiguous()

        cls = self.cls_token.clone().expand(B,self.cls_n,self.embed_dim)
        x = torch.cat([cls, x], dim=1)

        for blk, seq_blk in zip(self.blocks, self.seqblocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if t_token is not None:
                x = seq_blk(x, t_token)

        x = self.last_norm(x)

        cls = x[:,:self.cls_n]
        pos = torch.cat([self.root_proj(cls[:, 0]).unsqueeze(1), \
                         self.joint_proj(cls[:,1:]).view(B, -1, 2)], dim=1)

        return x, pos


class MultiClsViT(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_chans=3, num_classes=80, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0, norm_layer=None,
                 use_checkpoint=False, ratio=1, last_norm=True, cls_n=6):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.cls_n = cls_n

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.cls_token = nn.Parameter(torch.zeros(cls_n, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        self.conv_embed = nn.Sequential(ConvEmbed(7, in_chans, 64, 4, 2, norm_layer),
                                        nn.GELU(),
                                        ConvEmbed(3, 64, embed_dim, 2, 1, norm_layer))

        self.root_proj = nn.Linear(embed_dim, 2)
        self.joint_proj = nn.Linear(embed_dim, 8)

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_embed(x).view(B, self.embed_dim, -1).transpose(2, 1).contiguous()

        cls = self.cls_token.clone().expand(B, self.cls_n, self.embed_dim)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        cls = x[:,:self.cls_n]
        pos = torch.cat([self.root_proj(cls[:, 0]).unsqueeze(1), \
                         self.joint_proj(cls[:,1:]).view(B, -1, 2)], dim=1)

        return x, pos

