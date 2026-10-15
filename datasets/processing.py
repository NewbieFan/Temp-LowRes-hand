import torch
import numpy as np
import cv2
import random
from config import cfg
from PIL import Image


def convert2origin_pixel(uvd, inv_trans, H=224, W=224):
    '''
    uvd: B,J,3 joints uvd coord
    trans: B,2,3 img2bbox trans
    '''
    uv = (uvd[:,:,:2] + 1) * (cfg.input_img_shape[1] // 2)

    uv1 = torch.cat((uv[:, :, :2], torch.ones_like(uvd[:, :, :1])), dim=2)
    uv = (inv_trans @ uv1.transpose(-1, -2)).transpose(-1, -2)

    return uv

def pixel2camera(uvd, K):
    fx, fy, fu, fv = K[:, 0, 0:1], K[:, 1, 1:2], K[:, 0, 2:3], K[:, 1, 2:3]
    xyz = torch.zeros_like(uvd, dtype=torch.float32)
    xyz[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
    xyz[:, :, 1] = (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
    xyz[:, :, 2] = uvd[:, :, 2]
    return xyz



def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def procrustes_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(np.uint8(cv_img), cv2.COLOR_BGR2RGB))

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def load_img(path, order='BGR'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]
    y_img = y_img[joint_valid==1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin+xmax)/2.
    width = (xmax-xmin)*expansion_factor
    xmin = x_center - 0.5*width* 1.2
    xmax = x_center + 0.5*width* 1.2
    
    y_center = (ymin+ymax)/2.
    height = (ymax-ymin)*expansion_factor
    ymin = y_center - 0.5*height* 1.2
    ymax = y_center + 0.5*height* 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*expansion_factor
    bbox[3] = h*expansion_factor
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

def get_aug_config(do_flip):
    scale_factor = (0.9, 1.1)
    rot_factor = 180
    color_factor = 0.2
    transl_factor = 10
    scale = np.random.rand() * (scale_factor[1] - scale_factor[0]) + scale_factor[0]
    rot = (np.random.rand() * 2 - 1) * rot_factor
    transl_x = (np.random.rand() * 2 - 1) * transl_factor
    transl_y = (np.random.rand() * 2 - 1) * transl_factor
    transl = (transl_x, transl_y)
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    # if do_flip:
    #     do_flip = False
    # else:
    #     do_flip = random.random() <= 0.5

    return scale, rot, transl, color_scale, do_flip

def augment_with_config(img, bbox, split, config):
    if split.startswith('train'):
        scale, rot, transl, color_scale, do_flip = config
    else:
        scale, rot, transl, color_scale, do_flip = 1.0, 0.0, (0.0, 0.0), np.array([1, 1, 1]), config[-1]
    img, trans, inv_trans, trans_joint \
        = generate_patch_image(img, bbox, scale, rot, transl, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)

    return img, trans, inv_trans, rot, do_flip

def generate_patch_image(cvimg, bbox, scale, rot, transl, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl,
                                        inv=True)
    trans_joint = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], 1.0, 0.0, transl)

    return img_patch, trans, inv_trans, trans_joint

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, transl, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment translation
    src_center[0] += transl[0]
    src_center[1] += transl[1]

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


