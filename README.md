# Pose-Guided Temporal Enhancement for Robust Low-Resolution Hand Reconstruction [CVPR2025]

## Installation

```Shell
pip install -r requirements.txt
```

Tested with Python 3.9 and PyTorch 1.9, and with Python 3.10 and PyTorch 2.5
(cu121). Install a CUDA build of PyTorch for GPU training, for example
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

`chumpy` is required because the MANO model files are chumpy pickles, which also
pins `numpy` below 1.24 (`chumpy` and `imgaug` still use the removed `np.bool` /
`np.int` / `np.float` aliases). If building `chumpy` fails with
`No module named 'pip'`, install it separately with

```Shell
pip install --no-build-isolation chumpy
```

## MANO model files

`manopth` is included as a submodule, so clone this repository recursively (or
initialise the submodule afterwards):

```Shell
git clone --recursive https://github.com/NewbieFan/Temp-LowRes-hand.git
# if you already cloned without --recursive:
git submodule update --init
```

The MANO model itself is **not** redistributed here. Download `mano_v1_2.zip` from
<https://mano.is.tue.mpg.de>, then place the two hand models under the submodule,
where `utils/mano.py` expects them:

```Shell
manopth/mano/models/MANO_LEFT.pkl
manopth/mano/models/MANO_RIGHT.pkl
```

Set `MANO_ROOT` to load them from somewhere else.

## Data preparation

`tools/prepare_dexycb.py` builds the sequence annotations the dataloader reads.
Two modes:

```Shell
# make image paths in existing annotation files relative and store them gzipped
python tools/prepare_dexycb.py --rewrite-paths

# or rebuild the annotation files from the official DexYCB annotations
python tools/prepare_dexycb.py --from-official --dex-ycb-root /path/to/DexYCB
```

The `--from-official` mode reads `labels_*.npz` (`pose_m`, `joint_3d`, `joint_2d`),
`meta.yml` (`mano_sides`, `mano_calib`) and the camera and MANO calibration files
from a standard DexYCB download, and reproduces the shipped annotations frame by
frame. Frames without a hand annotation are skipped. By default every split of the
`s0` protocol is generated from all subjects present in the download, so a partial
download yields the corresponding part of each split; pass `--subjects` or
`--split` to restrict the run.

`tools/compare_annotations.py` checks a rebuild against the released annotations
field by field, optionally restricted to a subject prefix:

```Shell
python tools/compare_annotations.py datasets/DEX_YCB_s0_train_sequence.json.gz \
  /path/to/rebuilt.json.gz --prefix <subject>/
```

The rebuild reproduces `joint_3d`, `joint_2d`, `joint_3d_root`, `mano_pose`,
`mano_shape`, `mano_trans`, `img_path`, `img_shape`, `hand_type` and `cam_param`
exactly, and `bbox` to within one float32 ulp (3e-5 pixels).

Annotations are written to `datasets/DEX_YCB_s0_{train,val,test}_sequence.json.gz`;
both gzipped and plain `.json` files are accepted by the dataloader. `--rewrite-paths`
also converts an existing plain file to gzipped form and drops the unused `mesh_path`
field. The generated annotations are gitignored because they are large (about 700 MB
gzipped) and reproducible from the official dataset with `--from-official`.

DexYCB is expected under `--data-root` with its original layout:

```Shell
<data-root>/
├── <subject>/
│   └── <date>/
│       ├── 836212060125/
│       │   ├── color_000001.jpg
│       │   └── labels_000001.npz
│       └── meta.yml
├── ...
├── calibration/
└── models/
```

The released annotations follow the official `s0` split. Note that training
validates on the `s0` **test** split, matching the protocol used for the numbers
above.

## Training

```Shell
python train.py --data-root /path/to/DexYCB --output-root ./output
```

Optional flags: `--checkpoint` for pretrained weights, `--continue-train`,
`--epoch`, `--batch-size`, `--lr`, `--num-workers`, `--time-steps`, `--seq-gap`,
`--early-stop`, `--device`. Checkpoints and TensorBoard logs are written to
`<output-root>/<experiment>/{run,train}`.

`downsratio` in `config.py` sets the intermediate resolution the training images
are downsampled to before being resized back to `input_img_shape`. The released
default is `downsratio=64` with `input_img_shape=(64, 64)`, which leaves the
resolution unchanged; set `downsratio=32` for the harder 32x32 variant.

Two models are trained in sequence. `train_single.py` pretrains the backbone on
single frames and defaults to 20 epochs. It keeps only the joint tokens: six class
tokens go through `root_proj` / `joint_proj` to predict the 21 2D joint
coordinates, supervised with a smooth L1 loss, and no mesh head is involved.

```Shell
python train_single.py --data-root /path/to/DexYCB --output-root ./output \
  --checkpoint ./checkpoints/vitpose_base_hand.pth
```

`--checkpoint` takes the original weights of
[ViTPose-B](https://github.com/ViTAE-Transformer/ViTPose) directly, for example the
released hand model. The twelve transformer blocks are copied verbatim
(`backbone.blocks.*` to `blocks.*`) and ViTPose-B's single class token is expanded
to the six the model uses. The absolute position encoding is dropped rather than
interpolated, and the 16x16 patch embedding is replaced by the convolutional
embedding that produces the 8x8 token grid for 64x64 input, so that stem is trained
from scratch. The resulting `ckpt_best.pth` then seeds the sequential model through
`train.py --checkpoint`.

## Evaluation

Download the trained weights from
[Google Drive](https://drive.google.com/file/d/1zhlsBB0CaOAXtuFU16oD1NSuWIXDncGk/view?usp=drive_link)
and pass the file to `eval.py`:

```Shell
python eval.py --data-root /path/to/DexYCB --output-root ./output \
  --checkpoint /path/to/ckpt_best.pth
```

Without `--checkpoint` the weights are read from
`<output-root>/<experiment>/train/checkpoints/ckpt_best.pth`, and `--split` selects
another split. All seven metrics are logged to
`<output-root>/<experiment>/test/log/`:

```Shell
[MPJPE: 13.21]
[PA_MPJPE: 5.72]
[MPVPE: 12.73]
[PA_MPVPE: 5.52]
[AUC: 88.56]
[FAT5: 78.43]
[FAT15: 98.77]
```

Errors are in millimetres, MPJPE / MPVPE are root-relative, PA-* are Procrustes
aligned, AUC uses the PCK curve over 0-50 mm, and FAT5 / FAT15 are mesh F-scores at
5 mm and 15 mm.

## Citation

```bibtex
@INPROCEEDINGS{chao:cvpr2021,
  author    = {Yu-Wei Chao and Wei Yang and Yu Xiang and Pavlo Molchanov and Ankur Handa and Jonathan Tremblay and Yashraj S. Narang and Karl {Van Wyk} and Umar Iqbal and Stan Birchfield and Jan Kautz and Dieter Fox},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title     = {{DexYCB}: A Benchmark for Capturing Hand Grasping of Objects},
  year      = {2021},
}
```
