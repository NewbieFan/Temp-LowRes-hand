import os.path as osp
import copy
import gzip
import json
import re

import numpy as np
import torch
import cv2
import torchvision.transforms as standard
from scipy.spatial import cKDTree
import imgaug as ia
import imgaug.augmenters as iaa

from config import cfg
from manopth.manolayer import ManoLayer
from datasets.processing import load_img, procrustes_align, cam2pixel, \
    cv2pil, augment_with_config, get_aug_config
from utils.eval_util import EvalUtil
from utils.mano import MANO, MANO_ROOT

_SEQUENCE_JSON = 'DEX_YCB_s0_{}_sequence.json'
_SEQUENCE_KEY = re.compile(r'(20\d{6}-subject-\d{2}/.*)$')


def sequence_path(data_split):
    base = osp.join(osp.dirname(osp.abspath(__file__)), _SEQUENCE_JSON.format(data_split))
    for path in (base + '.gz', base):
        if osp.exists(path):
            return path
    return base + '.gz'


def load_sequence(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def resolve_img_path(img_path, root_dir):
    match = _SEQUENCE_KEY.search(img_path)
    if match is not None:
        img_path = match.group(1)
    if not osp.isabs(img_path):
        img_path = osp.join(root_dir, img_path)
    return img_path


def calculate_fscore(gt, pr, th=0.01):
    gt = np.asarray(gt, dtype=np.float64)
    pr = np.asarray(pr, dtype=np.float64)
    d1 = cKDTree(pr).query(gt, k=1)[0]
    d2 = cKDTree(gt).query(pr, k=1)[0]
    if len(d1) and len(d2):
        recall = float(np.count_nonzero(d2 < th)) / float(len(d2))
        precision = float(np.count_nonzero(d1 < th)) / float(len(d1))
        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


class DexYCB_Temporal(torch.utils.data.Dataset):
    def __init__(self, rootdir, data_split, T=5, gap=4):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])
        self.data_split = data_split
        self.root_dir = rootdir
        self.T = T
        self.gap = gap

        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.mano = MANO()
        self.manolayer = ManoLayer(center_idx=0,
                                    flat_hand_mean=False,
                                    ncomps=45,
                                    side='right',
                                    mano_root=MANO_ROOT,
                                    use_pca=True)

        self.manoleft = ManoLayer(flat_hand_mean=False,
                                ncomps=45,
                                side='left',
                                mano_root=MANO_ROOT,
                                use_pca=True)
        self.face = self.mano.face
        self.leftface = self.manoleft.th_faces.numpy()

        ia.seed(345)
        self.downsaug = cfg.downsaug
        if self.downsaug:
            self.augseq = iaa.Sequential([iaa.Resize(cfg.downsratio),
                                          iaa.Resize(cfg.input_img_shape),
                                          iaa.GaussianBlur(sigma=(0.1, 0.4))])

        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval = {
                'MPJPE' : [],
                'PA_MPJPE' : [],
                'MPVPE': [],
                'PA_MPVPE': [],
                'AUC':[],
                'FAT5':[],
                'FAT15':[]
            } 
            self.evalutil = EvalUtil(21)

    def __len__(self):
        return len(self.datalist)

    def load_data(self):
        db = load_sequence(sequence_path(self.data_split))
        datalist = []

        for seq, ann_list in db.items():
             for f in range(len(ann_list)):
                f_s = f - self.T*self.gap - 1
                f_s = None if f_s < 0 else f_s
                seq_data = ann_list[f: f_s: -self.gap][::-1]
                if self.data_split == "train" and len(seq_data) < self.T - 1:
                    continue
                if len(seq_data) < self.T + 1:
                    seq_data = (self.T+1 - len(seq_data))*seq_data[:1] + seq_data
                datalist.append(seq_data)

        print('{} samples in this split'.format(len(datalist)))

        return datalist

    def generate_norm_bbx(self, bbox):
        bb_c_x = torch.tensor(float(bbox[0] + 0.5 * bbox[2]))
        bb_c_y = torch.tensor(float(bbox[1] + 0.5 * bbox[3]))
        bb_width = torch.tensor(float(bbox[2]))
        bb_height = torch.tensor(float(bbox[3]))
        return torch.stack([bb_c_x, bb_c_y, bb_width, bb_height], dim=0)

    def get_single(self, seq_data, idx, aug_config):
        data = copy.deepcopy(seq_data[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')
        scale, rot, transl, color_scale, _ = aug_config
        aug_config = (scale, rot, transl, color_scale, do_flip)

        ## img
        img_path = resolve_img_path(img_path, self.root_dir)
        img = load_img(img_path, 'BGR')

        ## img augment && cropping
        img, img2bb_trans, bb2img_trans, rot, _= \
            augment_with_config(img, bbox, self.data_split, aug_config)
        if self.downsaug:
            img = self.augseq(images=[img])[0]
        img = self.transform(cv2pil(img))
        bbox = self.generate_norm_bbx(bbox)

        intrinsics = data['cam_param']
        K = [[intrinsics['focal'][0], 0., intrinsics['princpt'][0]],
             [0., intrinsics['focal'][1], intrinsics['princpt'][1]],
             [0., 0., 1.]]

        ## joint3d and mesh3d
        joint_3d = np.float32(data['joint_3d'])
        root_joint_cam = copy.deepcopy(joint_3d[self.root_joint_idx])

        ## 读取Mesh信息
        mano_pose, mano_shape, mano_trans = \
               np.float32(data['mano_pose']), np.float32(data['mano_shape']), np.float32(data['mano_trans'])
        th_pose = torch.from_numpy(mano_pose)[None]
        th_shape = torch.from_numpy(mano_shape)[None]
        th_trans = torch.from_numpy(mano_trans)[None]
        if do_flip:
            mesh_xyz, kp_xyz = self.manoleft(th_pose, th_shape, th_trans)
        else:
            mesh_xyz, kp_xyz = self.manolayer(th_pose, th_shape, th_trans)

        mesh_xyz = mesh_xyz[0] / 1000.
        mesh_xyz = mesh_xyz.cpu().numpy()

        mesh_3d = mesh_xyz.copy()
        joint_xyz = joint_3d.copy() 

        mesh_pose_xyz = np.concatenate((mesh_xyz, joint_3d), axis=0)
        mesh_pose_uvd = cam2pixel(mesh_pose_xyz, intrinsics['focal'], intrinsics['princpt'])
        if do_flip:
            mesh_pose_uvd[:,0] = img_shape[1] - mesh_pose_uvd[:,0] - 1

        uvd_padding = np.concatenate((mesh_pose_uvd[:, :2], np.ones_like(mesh_pose_uvd[:, :1])), 1)
        mesh_pose_uvd[:,:2] = np.dot(img2bb_trans, uvd_padding.transpose(1, 0)).transpose(1, 0)[:, :2]
        mesh_pose_uvd[:, 0] = mesh_pose_uvd[:,0] / (cfg.input_img_shape[1]//2) -1
        mesh_pose_uvd[:, 1] = mesh_pose_uvd[:,1] / (cfg.input_img_shape[0]//2) -1

        center_depth = np.mean(mesh_pose_uvd[:, -1])

        guide_pose = mesh_pose_uvd[cfg.num_vert:].copy()
        guide_pose[:, 2:3] = (guide_pose[:, 2:3] - center_depth) / (cfg.depth_box / 2)

        mesh_pose_uvd[:, 2:3] = (mesh_pose_uvd[:, 2:3] - root_joint_cam[None, 2:3]) / cfg.depth_box

        ## mano parameter   
        mano_pose = np.float32(mano_pose).reshape(-1, 3)
        joint_3d -= root_joint_cam  # root-relative
        mesh_xyz -= root_joint_cam  # root-relative
        if do_flip:
            joint_3d[:, 0] *= -1
            mesh_xyz[:, 0] *= -1
            mano_pose[:, 1:] *= -1

        if self.data_split == 'train':
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joint_3d = np.dot(rot_aug_mat, joint_3d.transpose(1, 0)).transpose(1, 0)
            mesh_xyz = np.dot(rot_aug_mat, mesh_xyz.transpose(1, 0)).transpose(1, 0)

            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

        else:  # test and val
            mano_pose = mano_pose.reshape(-1)
            rot_aug_mat = np.eye(3, dtype=np.float32)

        inputs = {'img': np.float32(img)}

        targets = {'mesh_xyz': torch.from_numpy(mesh_3d).float(),
                   'joint_xyz': torch.from_numpy(joint_xyz).float(),
                   'guide_pose': torch.from_numpy(guide_pose).float(),
                   'mesh_pose_uvd': torch.from_numpy(mesh_pose_uvd).float()}
                   
        meta_info = {
            'img_shape':np.float32(img_shape),
            'do_flip': np.float32(1) if do_flip else np.float32(0),
            'joint_3d_root': np.float32(root_joint_cam),
            'cam_focal':  np.float32(data['cam_param']['focal']),
            'cam_princpt':  np.float32(data['cam_param']['princpt']),
            'img2bb_trans': np.float32(img2bb_trans),
            'bb2img_trans': np.float32(bb2img_trans),
            'root_depth': np.float32(root_joint_cam[-1]),
            'K': np.float32(K)
        }
        return inputs, targets, meta_info

    def __getitem__(self, idx):
        aug_config = get_aug_config(False)
        seq_data = copy.deepcopy(self.datalist[idx])
        input, target, meta_info = self.get_single(seq_data, 0, aug_config)
        img_seq = {k: [v[None]] for k,v in input.items()}
        targets = {k: [v[None]] for k,v in target.items()}
        meta_infos = {k: [v[None]] for k,v in meta_info.items()}

        for it in range(1, self.T+1):
            input, target, meta_info = \
                self.get_single(seq_data, it, aug_config)
            img_seq['img'].append(input['img'][None])
            for k, v in meta_info.items():
                meta_infos[k].append(v[None])
            for k,v in target.items():
                targets[k].append(v[None])
        img_seq = {'img': np.concatenate(img_seq['img'])}
        meta_infos = {k:np.concatenate(v) for k,v in meta_infos.items()}
        targets = {k:np.concatenate(v) for k,v in targets.items()}
        return img_seq, targets, meta_infos

    def test(self, preds, f_id, cur_sample_idx, target=None):
        batch_size = len(preds['pose_out'])
        for n in range(batch_size):
            pred_joints = np.array(preds['pose_out'][n])
            pred_mesh = np.array(preds['mesh_out'][n])
            idx = cur_sample_idx + n

            gt_joints = copy.deepcopy(self.datalist[idx][f_id]['joint_3d'])
            gt_joints = np.float32(gt_joints)

            if target is not None:
                mesh_xyz = target['mesh_xyz'][n]
            else:
                data = copy.deepcopy(self.datalist[idx][f_id])
                mano_pose, mano_shape, mano_trans = \
                    np.float32(data['mano_pose']), np.float32(data['mano_shape']), np.float32(data['mano_trans'])
                hand_type = data['hand_type']
                do_flip = (hand_type == 'left')
                th_pose = torch.from_numpy(mano_pose)[None]
                th_shape = torch.from_numpy(mano_shape)[None]
                th_trans = torch.from_numpy(mano_trans)[None]
                if do_flip:
                    mesh_xyz, kp_xyz = self.manoleft(th_pose, th_shape, th_trans)
                else:
                    mesh_xyz, kp_xyz = self.manolayer(th_pose, th_shape, th_trans)
                mesh_xyz = mesh_xyz[0] / 1000.
                mesh_xyz = mesh_xyz.cpu().numpy()

            # root_relative
            pred_mesh -=  pred_joints[0:1]
            pred_joints = pred_joints - pred_joints[0:1]

            mesh_xyz -= gt_joints[0:1]
            gt_joints = gt_joints - gt_joints[0:1]

            joints_out_aligned_procrustes = procrustes_align(pred_joints, gt_joints.copy())
            mesh_out_aligned_procrustes = procrustes_align(pred_mesh, mesh_xyz.copy())

            # m -> cm
            pred_joints *= 1000
            gt_joints *= 1000
            pred_mesh *= 1000
            mesh_xyz *= 1000
            joints_out_aligned_procrustes *= 1000
            mesh_out_aligned_procrustes *= 1000

            if self.data_split == 'test':
                f_5, _, _ = calculate_fscore(mesh_xyz.copy(), mesh_out_aligned_procrustes.copy(), 5)
                f_15, _, _ = calculate_fscore(mesh_xyz.copy(), mesh_out_aligned_procrustes.copy(), 15)
                self.eval['FAT5'].append(f_5 * 100.)
                self.eval['FAT15'].append(f_15 * 100.)

                self.evalutil.feed(gt_joints.copy(), 0., joints_out_aligned_procrustes.copy(), True)

            self.eval['MPJPE'].append(np.sqrt(np.sum((pred_joints - gt_joints) ** 2, 1)).mean())
            self.eval['PA_MPJPE'].append(np.sqrt(np.sum((joints_out_aligned_procrustes - gt_joints) ** 2, 1)).mean())
            self.eval['MPVPE'].append(np.sqrt(np.sum((pred_mesh - mesh_xyz)**2, 1)).mean())
            self.eval['PA_MPVPE'].append(np.sqrt(np.sum((mesh_out_aligned_procrustes - mesh_xyz)**2, 1)).mean())

    def get_total_result(self):
        epe_mean, epe_median, auc, pck, _ = self.evalutil.get_measures(0.0, 50.0, 100)
        self.eval['AUC'].append(auc*100)
        return {k:np.mean(v) for k,v in self.eval.items()}

    def clean_result(self):
        self.eval = {
            'MPJPE' : [],
            'PA_MPJPE' : [],
            'MPVPE': [],
            'PA_MPVPE': [],
            'AUC':[],
            'FAT5':[],
            'FAT15':[]
        }


if __name__ == '__main__':
    import random
    dataset = DexYCB_Temporal(cfg.data_root, 'val')
    ii = dataset.__getitem__(random.randint(0, len(dataset)))
    inputs, targets, meta_infos = ii
    for k, v in inputs.items():
        print(k, np.shape(v))
    for k, v in targets.items():
        print(k, np.shape(v))
