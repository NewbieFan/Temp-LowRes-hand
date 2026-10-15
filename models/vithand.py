import torch
import torch.nn as nn
import torch.nn.functional as F


from .backbone.temp_vit import MultiClsViT, MultiClsSeqViTv2, SpatioTemporalModule

from .losses.hand_loss import NormalVectorLoss, EdgeLengthLoss

import configs.sar_dexycb_64x64 as hand_cfg_64

from config import cfg
from utils.mano import MANO

class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size[0], 1).view([1, self.size[0]]).repeat([self.size[1], 1])
        self.wy = torch.arange(0.0, 1.0 * self.size[1], 1).view([self.size[1], 1]).repeat([1, self.size[0]])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)
        return keypoint_uv, scoremap

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj = nn.Parameter(torch.eye(num_joint).float().to(cfg.device), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class SAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, num_vert, template):
        super(SAIGB, self).__init__()
        self.template = torch.Tensor(template).to(cfg.device)  # self.mano.template
        self.backbone_channels = backbone_channels
        self.num_vert = num_vert
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_vert, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        feature_size = x.shape[-1]*x.shape[-2]
        feature = self.group(x).view(-1, self.num_vert, feature_size * self.num_FMs)
        template = self.template.repeat(x.shape[0], 1, 1)
        init_graph = torch.cat((feature, template), dim=2)
        return init_graph

class GBBMR(nn.Module):
    def __init__(self, in_dim, num_vert, num_joint, heatmap_shape):
        super(GBBMR, self).__init__()
        self.in_dim = in_dim
        self.num_vert = num_vert
        self.num_joint = num_joint
        self.num_total = num_vert + num_joint
        self.heatmap_shape = heatmap_shape
        heatmap_size = self.heatmap_shape[0] * self.heatmap_shape[1]
        self.soft_heatmap = SoftHeatmap(self.heatmap_shape, self.num_total)
        self.reg_xy = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, heatmap_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, heatmap_size, heatmap_size),
        )
        self.reg_z = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, heatmap_size), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, heatmap_size, heatmap_size),
        )
        self.mesh2pose_hm = nn.Linear(self.num_vert, self.num_joint)
        self.mesh2pose_dm = nn.Linear(self.num_vert, self.num_joint)

    def forward(self, x):
        init_graph = x
        heatmap_xy_mesh = self.reg_xy(init_graph).view(-1, self.num_vert, self.heatmap_shape[1], self.heatmap_shape[0])
        heatmap_z_mesh = self.reg_z(init_graph).view(-1, self.num_vert, self.heatmap_shape[1], self.heatmap_shape[0])
        heatmap_xy_joint = self.mesh2pose_hm(heatmap_xy_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_z_joint = self.mesh2pose_dm(heatmap_z_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_xy = torch.cat((heatmap_xy_mesh, heatmap_xy_joint), dim=1)
        heatmap_z = torch.cat((heatmap_z_mesh, heatmap_z_joint), dim=1)
        coord_xy, latent_heatmaps = self.soft_heatmap(heatmap_xy) 
        depth_maps = latent_heatmaps * heatmap_z
        coord_z = torch.sum(
            depth_maps.view(-1, self.num_total, depth_maps.shape[2] * depth_maps.shape[3]), dim=2, keepdim=True)
        joint_coord = torch.cat((coord_xy, coord_z), 2)
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / (self.heatmap_shape[0] // 2) - 1
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / (self.heatmap_shape[1] // 2) - 1
        return joint_coord, latent_heatmaps, depth_maps

class SARhead(nn.Module):
    def __init__(self, 
                 in_channels=768, 
                 num_FMs=8, 
                 num_joints=21, 
                 num_verts=778, 
                 feature_shape=[16,16], 
                 heatmap_shape=[32,32],):
        super().__init__()
        mano = MANO()

        self.feature_shape = feature_shape
        self.feature_size = feature_shape[0] * feature_shape[1]

        self.saigb = SAIGB(in_channels, num_FMs, num_verts, mano.template)
        self.gbbmr = GBBMR(num_FMs*self.feature_size+3, num_verts, num_joints, heatmap_shape)

        self.coord_loss = nn.L1Loss()
        self.normal_loss = NormalVectorLoss(mano.face)
        self.edge_loss = EdgeLengthLoss(mano.face)

    def forward(self, x, target=None):
        '''x: (B,C,Hp,Wp)'''
        B = x.shape[0]
        features = x.permute(0,2,1).view(B,-1,self.feature_shape[0],self.feature_shape[1]).contiguous()

        init_graph = self.saigb(features)
        coord, lhm, dm = self.gbbmr(init_graph)

        if self.training and target is not None:
            loss = {}
            mesh_pose_uvd = target['mesh_pose_uvd'].to(cfg.device)
            loss['coord_loss'] = self.coord_loss(coord, mesh_pose_uvd)
            loss['normal_loss'] = \
                self.normal_loss(coord[:, :cfg.num_vert], mesh_pose_uvd[:, :cfg.num_vert]).mean() * 0.1
            loss['edge_loss'] = \
                self.edge_loss(coord[:, :cfg.num_vert], mesh_pose_uvd[:, :cfg.num_vert]).mean()
            return loss
        else:
            return coord


class vithand(nn.Module):
    def __init__(self, h_cfg:dict):
        super(vithand, self).__init__()
        
        backbone_cfg = {k: v for k, v in h_cfg['backbone'].items() if k != 'type'}
        
        self.backbone = MultiClsViT(**backbone_cfg)

    def forward(self, x):
        return self.backbone(x)


class SarSeqViThand(nn.Module):
    def __init__(self, h_cfg: dict, time_steps=5, cls_n=6):
        super().__init__()
        self.cls_n = cls_n

        backbone_cfg = {k: v for k, v in h_cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in h_cfg['keypoint_head'].items() if k != 'type'}
        
        self.backbone = MultiClsSeqViTv2(**backbone_cfg)
        self.keypoint_head = SARhead(**head_cfg)

        self.coarse_joints = []
        def joint_hook(_, input, output):
            self.coarse_joints.append(output)
        for name, module in self.backbone.seqblocks.named_modules():
            if 'pos_hook' in name:
                module.register_forward_hook(joint_hook)
        
        self.time_steps = time_steps
        self.token_dim = h_cfg['backbone']['embed_dim']
        self.feat_shape = h_cfg['keypoint_head']['feature_shape']
        self.feat_size = self.feat_shape[0] * self.feat_shape[1]
        num_heads = h_cfg['backbone']['num_heads']

        self.STModule = nn.Sequential(SpatioTemporalModule(self.token_dim, num_heads),
                                      SpatioTemporalModule(self.token_dim, num_heads),
                                      SpatioTemporalModule(self.token_dim, num_heads))

    def pre_forward(self, x):
        B, T, C, H, W = x.shape

        outs, joints = self.backbone(x.view(-1, C, H, W)) 

        features = outs[:,self.cls_n:].permute(0,2,1).view(B*T,-1,self.feat_shape[0],self.feat_shape[1]).contiguous()
        temporal_joints_token = F.grid_sample(features, joints[:,None])

        temporal_joints_token = temporal_joints_token.view(B*T,self.token_dim,-1).permute(0,2,1).contiguous()

        cls = outs[:,:self.cls_n]
        cls_ = torch.cat([cls[:,:1], cls[:,1:].repeat_interleave(4,dim=1)], dim=1)
        temporal_joints_token = temporal_joints_token + cls_

        temporal_joints_token = temporal_joints_token.view(B, T, -1, self.token_dim)

        return temporal_joints_token, joints.view(B, T, -1, 2)
    
    def forward(self, x, t, temporal_tokens, target=None):
        self.coarse_joints = []      
        if temporal_tokens.shape[1] > self.time_steps:
            temporal_tokens = temporal_tokens[:,:self.time_steps]

        temporal_tokens = self.STModule(temporal_tokens)

        B, _, H, W = x.shape
        feats, jt_uv = self.backbone(x, temporal_tokens)

        return self.keypoint_head(feats[:,self.cls_n:], target), temporal_tokens, jt_uv


def getSarMultiSeqModel():
    return SarSeqViThand(hand_cfg_64.model)


def getJointTokenModel():
    return vithand(hand_cfg_64.model)


def load_vitpose_b(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'network' in ckpt:
        state_dict = ckpt['network']
    else:
        state_dict = ckpt

    mapped = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.blocks.'):
            mapped[k] = v
        elif k == 'backbone.cls_token':
            mapped[k] = v.reshape(1, -1).expand(model.backbone.cls_n, -1).clone()

    return model.load_state_dict(mapped, strict=False)

if __name__ == '__main__':
    pass