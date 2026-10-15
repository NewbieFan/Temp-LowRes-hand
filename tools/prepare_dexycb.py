import argparse
import gzip
import json
import os
import os.path as osp
import re
import sys
from glob import glob

import numpy as np
import yaml

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from datasets.processing import get_bbox, process_bbox

DATASET_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'datasets')

SEQUENCE_JSON = 'DEX_YCB_s0_{}_sequence.json'
SEQUENCE_KEY = re.compile(r'(20\d{6}-subject-\d{2}/.*)$')

SPLITS = ['train', 'val', 'test']
SUBJECTS = [
    '20200709-subject-01', '20200813-subject-02', '20200820-subject-03',
    '20200903-subject-04', '20200908-subject-05', '20200918-subject-06',
    '20200928-subject-07', '20201002-subject-08', '20201015-subject-09',
    '20201022-subject-10',
]
SERIALS = [
    '836212060125', '839512060362', '840412060917', '841412060263',
    '932122060857', '932122060861', '932122061900', '932122062010',
]
S0_SUBJECT_IND = {'train': list(range(10)), 'val': [0, 1], 'test': list(range(2, 10))}
S0_SEQUENCE_IND = {
    'train': [i for i in range(100) if i % 5 != 4],
    'val': [i for i in range(100) if i % 5 == 4],
    'test': [i for i in range(100) if i % 5 == 4],
}

IMG_SHAPE = [480, 640]
COLOR_FORMAT = 'color_{:06d}.jpg'
LABEL_FORMAT = 'labels_{:06d}.npz'


def sequence_path(split, out_dir):
    base = osp.join(out_dir, SEQUENCE_JSON.format(split))
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


def dump_sequence(db, path):
    if path.endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(db, f)
    else:
        with open(path, 'w') as f:
            json.dump(db, f)


def to_f32(value):
    return np.asarray(value, dtype=np.float32).tolist()


def rewrite_paths(out_dir):
    for split in SPLITS:
        src = sequence_path(split, out_dir)
        db = load_sequence(src)
        count = 0
        for seq, frames in db.items():
            for frame in frames:
                match = SEQUENCE_KEY.search(frame['img_path'])
                if match is not None:
                    frame['img_path'] = match.group(1)
                    count += 1
                frame.pop('mesh_path', None)
        dst = osp.join(out_dir, SEQUENCE_JSON.format(split)) + '.gz'
        dump_sequence(db, dst)
        if src != dst and osp.exists(src):
            os.remove(src)
        print('{}: {} sequences, {} image paths rewritten -> {}'.format(
            split, len(db), count, osp.basename(dst)))


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_intrinsics(dex_ycb_root, serial):
    path = osp.join(dex_ycb_root, 'calibration', 'intrinsics',
                    '{}_640x480.yml'.format(serial))
    intr = load_yaml(path)['color']
    return {'focal': [intr['fx'], intr['fy']], 'princpt': [intr['ppx'], intr['ppy']]}


def read_betas(dex_ycb_root, mano_calib):
    path = osp.join(dex_ycb_root, 'calibration', 'mano_{}'.format(mano_calib), 'mano.yml')
    return load_yaml(path)['betas']


def build_sequence(dex_ycb_root, split, out_dir, only_subjects=None):
    subjects = [SUBJECTS[i] for i in S0_SUBJECT_IND[split]]
    if only_subjects:
        subjects = [s for s in subjects if s in only_subjects]
    sequence_ind = S0_SEQUENCE_IND[split]
    db = {}

    for subject in subjects:
        subject_dir = osp.join(dex_ycb_root, subject)
        if not osp.isdir(subject_dir):
            print('skip missing subject {}'.format(subject))
            continue
        dates = sorted(d for d in os.listdir(subject_dir)
                       if osp.isdir(osp.join(subject_dir, d)))
        if len(dates) != 100:
            raise SystemExit('{}: expected 100 sequences, found {} - is the '
                             'download complete?'.format(subject, len(dates)))
        dates = [dates[i] for i in sequence_ind]
        for date in dates:
            meta = load_yaml(osp.join(dex_ycb_root, subject, date, 'meta.yml'))
            hand_type = meta['mano_sides'][0]
            betas = read_betas(dex_ycb_root, meta['mano_calib'][0])

            for serial in SERIALS:
                cam_dir = osp.join(dex_ycb_root, subject, date, serial)
                if not osp.isdir(cam_dir):
                    continue
                cam_param = read_intrinsics(dex_ycb_root, serial)
                frames = []
                for label_path in sorted(glob(osp.join(cam_dir, 'labels_*.npz'))):
                    frame_idx = int(osp.basename(label_path)[len('labels_'):-len('.npz')])
                    labels = np.load(label_path)
                    pose_m = np.asarray(labels['pose_m'][0], dtype=np.float32)
                    joint_3d = np.asarray(labels['joint_3d'][0], dtype=np.float32)
                    joint_2d = np.asarray(labels['joint_2d'][0], dtype=np.float32)

                    if (joint_3d == -1).all():
                        continue

                    bbox = get_bbox(joint_2d[:, :2], np.ones_like(joint_2d[:, 0]),
                                    expansion_factor=1.5)
                    bbox = process_bbox(bbox, IMG_SHAPE[1], IMG_SHAPE[0],
                                        expansion_factor=1.0)
                    if bbox is None:
                        continue

                    frames.append({
                        'img_path': '/'.join([subject, date, serial,
                                              COLOR_FORMAT.format(frame_idx)]),
                        'img_shape': IMG_SHAPE,
                        'bbox': to_f32(bbox),
                        'hand_type': hand_type,
                        'cam_param': cam_param,
                        'joint_3d': to_f32(joint_3d),
                        'joint_2d': to_f32(joint_2d),
                        'joint_3d_root': to_f32(joint_3d[0]),
                        'mano_pose': to_f32(pose_m[:48]),
                        'mano_shape': to_f32(betas),
                        'mano_trans': to_f32(pose_m[48:51]),
                    })
                if len(frames) == 0:
                    continue
                key = '/'.join([subject, date, serial])
                db[key] = frames
                print('  {} frames in {}'.format(len(frames), key))

    path = osp.join(out_dir, SEQUENCE_JSON.format(split)) + '.gz'
    dump_sequence(db, path)
    print('{}: {} sequences, {} frames -> {}'.format(
        split, len(db), sum(len(v) for v in db.values()), path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewrite-paths', action='store_true',
                        help='rewrite absolute image paths in the shipped json files')
    parser.add_argument('--from-official', action='store_true',
                        help='rebuild the json files from official DexYCB annotations')
    parser.add_argument('--dex-ycb-root', default=os.environ.get('DEX_YCB_DIR', ''))
    parser.add_argument('--split', default=None, choices=SPLITS + [None])
    parser.add_argument('--subjects', default=None,
                        help='comma separated subject directory names to rebuild')
    parser.add_argument('--out-dir', default=DATASET_DIR)
    args = parser.parse_args()

    if not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.rewrite_paths:
        rewrite_paths(args.out_dir)

    if args.from_official:
        if not osp.isdir(args.dex_ycb_root):
            raise SystemExit('--dex-ycb-root is not a directory: {}'.format(args.dex_ycb_root))
        only = args.subjects.split(',') if args.subjects else None
        for split in ([args.split] if args.split else SPLITS):
            build_sequence(args.dex_ycb_root, split, args.out_dir, only)
