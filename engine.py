import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from datasets.processing import convert2origin_pixel, pixel2camera


def postprocess(outs, meta_infos):
    pred_uv = convert2origin_pixel(outs, meta_infos['bb2img_trans'][:,-1].to(cfg.device))
    preds = torch.cat([pred_uv, outs[:,:,2:3]], dim=-1)
    preds[:, :, -1] = preds[:, :, -1] * cfg.depth_box + meta_infos['root_depth'][:,-1:].to(cfg.device)

    do_flip = meta_infos['do_flip'][:,-1:].to(cfg.device)
    img_width = meta_infos['img_shape'][:,-1:,-1].to(cfg.device)
    pred_u = preds[:,:,0]
    preds[:,:,0] = (img_width - pred_u - 1) * do_flip + pred_u * (1 - do_flip)

    preds = pixel2camera(preds, meta_infos['K'][:,-1].to(cfg.device))
    return preds.cpu().numpy()


@torch.no_grad()
def predict(model, inputs, meta_infos):
    temporal_tokens, t_joints = \
        model.pre_forward(torch.flip(inputs['img'][:,:-1], [1]).to(cfg.device))

    img = inputs['img'][:,-1].to(cfg.device)
    outs, temporal_tokens, _ = model(img, cfg.time_steps, temporal_tokens, t_joints)

    return postprocess(outs, meta_infos)


@torch.no_grad()
def predict_joints(model, inputs, meta_infos):
    _, pos = model(inputs['img'][:,-1].to(cfg.device))
    inv_trans = meta_infos['bb2img_trans'][:,-1].to(cfg.device)
    return convert2origin_pixel(pos, inv_trans).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, dataset, predictor=predict):
    model.eval()
    dataset.clean_result()
    cur_sample_idx = 0
    for inputs, targets, meta_infos in tqdm(loader):
        preds = predictor(model, inputs, meta_infos)
        target = {k: v[:,-1].cpu().numpy() for k, v in targets.items()}
        dataset.test({'pose_out': preds[:, cfg.num_vert:],
                      'mesh_out': preds[:, :cfg.num_vert]},
                     -1, cur_sample_idx, target)
        cur_sample_idx += inputs['img'].shape[0]

    return dataset.get_total_result()


@torch.no_grad()
def evaluate_joints(model, loader):
    model.eval()
    errs, hits = [], []
    for inputs, targets, meta_infos in tqdm(loader):
        inv_trans = meta_infos['bb2img_trans'][:,-1].to(cfg.device)
        pred = predict_joints(model, inputs, meta_infos)
        gt = convert2origin_pixel(targets['guide_pose'][:,-1,:,:2].to(cfg.device),
                                  inv_trans).cpu().numpy()
        dist = np.linalg.norm(pred - gt, axis=-1)
        errs.append(dist.mean())
        hits.append((dist < 5).mean())

    return {'JointErr': float(np.mean(errs)), 'PCK@5': float(np.mean(hits)) * 100}
