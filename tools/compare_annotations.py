import argparse
import gzip
import json
import sys

import numpy as np

FLOAT_FIELDS = ['bbox', 'joint_3d', 'joint_2d', 'joint_3d_root',
                'mano_pose', 'mano_shape', 'mano_trans']
EXACT_FIELDS = ['img_path', 'img_shape', 'hand_type', 'cam_param']


def load(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compare two DexYCB sequence annotation files field by field')
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--prefix', default='',
                        help='only compare sequences starting with this prefix')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='tolerance for float fields, default one float32 ulp at 256')
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)

    ref_keys = sorted(k for k in reference if k.startswith(args.prefix))
    cand_keys = sorted(k for k in candidate if k.startswith(args.prefix))
    print('prefix             :', args.prefix or '(all)')
    print('reference sequences:', len(ref_keys))
    print('candidate sequences:', len(cand_keys))
    missing = [k for k in ref_keys if k not in candidate]
    extra = [k for k in cand_keys if k not in reference]
    print('missing in candidate:', len(missing), missing[:3])
    print('extra in candidate  :', len(extra), extra[:3])

    field_err = {f: 0.0 for f in FLOAT_FIELDS}
    exact_bad = {f: 0 for f in EXACT_FIELDS}
    frames = 0
    count_bad = 0
    for key in ref_keys:
        if key not in candidate:
            continue
        a, b = reference[key], candidate[key]
        if len(a) != len(b):
            count_bad += 1
            print('  frame count mismatch', key, len(a), len(b))
            continue
        for fa, fb in zip(a, b):
            frames += 1
            for f in FLOAT_FIELDS:
                d = float(np.abs(np.asarray(fa[f], np.float64) -
                                 np.asarray(fb[f], np.float64)).max())
                if d > field_err[f]:
                    field_err[f] = d
            for f in EXACT_FIELDS:
                if fa[f] != fb[f]:
                    exact_bad[f] += 1

    print('frames compared     :', frames)
    print('frame count mismatch:', count_bad)
    print('--- float fields (max abs diff) ---')
    for f in FLOAT_FIELDS:
        print('   %-14s %.3e' % (f, field_err[f]))
    print('--- exact fields (mismatch count) ---')
    for f in EXACT_FIELDS:
        print('   %-14s %d' % (f, exact_bad[f]))

    ok = (not missing and not extra and count_bad == 0
          and all(v <= args.tol for v in field_err.values())
          and all(v == 0 for v in exact_bad.values()))
    print()
    print('MATCH' if ok else 'DIFFERS - see above')
    sys.exit(0 if ok else 1)
