# LongStream: Long-Sequence Streaming Autoregressive Visual Geometry [CVPR 2026]

<p align="center">
  <a href="https://huggingface.co/spaces/NicolasCC/LongStream"><img src="https://img.shields.io/badge/Demo-Hugging_Face-yellow?style=for-the-badge" alt="Demo"></a>
  <a href="https://huggingface.co/NicolasCC/LongStream"><img src="https://img.shields.io/badge/Model-Hugging_Face-orange?style=for-the-badge" alt="Model"></a>
  <a href="https://3dagentworld.github.io/longstream/"><img src="https://img.shields.io/badge/Project_Page-Website-blue?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2602.13172"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" alt="Paper"></a>
</p>

<p align="center">
  <a href="examples/teaser.mp4">
    <img src="examples/teaser.gif" alt="LongStream teaser" width="960">
  </a>
</p>

## Abstract

Long-sequence streaming 3D reconstruction remains difficult because autoregressive visual geometry models usually anchor all poses to the first frame, which makes long-horizon prediction increasingly unstable and prone to scale drift. LongStream addresses this with a gauge-decoupled streaming formulation that predicts keyframe-relative poses, disentangles metric scale learning from geometry prediction, and periodically refreshes streaming caches to suppress long-term degradation. The resulting system supports stable metric-scale pose, depth, and point-cloud reconstruction across hundreds to thousands of frames, while remaining practical for release with full inference, evaluation, plotting, and interactive demo tooling.

## ToDoList

- [x] Weights release
- [x] Model inference script
- [x] Minimal CLI
- [x] Evaluation script
- [x] Plotting utilities
- [x] Interactive demo
- [ ] Data processing scripts release. Waiting for company approval.
- [ ] Training scripts and training code release. Waiting for company approval.

## Installation

Create a clean conda environment and install the runtime dependencies:

```bash
conda create -n longstream python=3.10 -y
conda activate longstream
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

Notes:

- `ffmpeg` is required for RGB and depth video export.
- Please follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install Torch and CUDA versions compatible with your hardware and drivers.
- Sky masking is enabled by default in the release config. It requires `onnxruntime` and may auto-download `skyseg.onnx` on first use; for offline runs or to silence this fallback path, use `--no-mask-sky`.

## Dataset Format and Conventions

LongStream expects scene data in the `generalizable` layout. KITTI, VKITTI 2, and Waymo should be downloaded from their official sources and converted or reorganized into this layout before running inference. Private datasets can be used directly as long as they follow the same structure and camera convention.

Official dataset links:

- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)
- [Waymo Open Dataset](https://waymo.com/open/download/)

### Generalizable Multi-Camera Layout

```text
<meta_root>/
  data_roots.txt
  <scene_name>/
    images/
      <camera_id>/
        000000.png|jpg
        000001.png|jpg
        ...
    cameras/
      <camera_id>/
        intri.yml
        extri.yml
    depths/
      <camera_id>/
        000000.exr
        000001.exr
        ...
```

Ground-truth camera files and depth files are optional for inference. Missing ground truth only disables the corresponding evaluation metrics.

### Pose and Camera Convention

LongStream uses the following convention consistently for both input annotations and saved outputs:

- `extri.yml` and `poses/abs_pose.txt` store `w2c` extrinsics.
- `intri.yml` and `poses/intri.txt` follow the OpenCV pinhole camera convention.
- Camera axes follow OpenCV: `x` right, `y` down, `z` forward.
- Depth maps are metric depths along the positive camera `z` axis.
- The world frame can be arbitrary, but every camera and depth map inside one scene must share the same world frame.

### Dataset Conversion

KITTI odometry:

- Download the official KITTI odometry sequences.
- Point `--src` to the KITTI root containing `sequences/<seq>/image_2` and optionally `image_3`.
- Convert into the generalizable meta-root:

```bash
python scripts/kitti_to_generalizable.py \
  --src /path/to/kitti_odometry_root \
  --out /path/to/meta_root
```

Waymo:

- Export each scene into the generalizable scene layout first, with `images/`, `cameras/`, and optional `depths/`.
- `scripts/waymo_to_generalizable.py` is a reorganization helper for an already-exported Waymo meta-root. It does not parse raw Waymo TFRecords directly.

```bash
python scripts/waymo_to_generalizable.py \
  --src /path/to/waymo_meta_root \
  --out /path/to/meta_root
```

VKITTI 2:

- Download from the official Virtual KITTI 2 page above.
- Export images, intrinsics, extrinsics, and optional depths into the same generalizable layout shown above.

Private datasets:

- Export images, intrinsics, extrinsics, and optional depth directly into the generalizable layout.
- No extra conversion script is required once the layout and camera convention match the specification above.

## Checkpoints

Download the released checkpoint from the [Hugging Face model repo](https://huggingface.co/NicolasCC/LongStream).

You can use it in either of these ways:

- Place `50_longstream.pt` at `checkpoints/50_longstream.pt`.
- Or run directly from Hugging Face:

  When `--hf-repo` or `--hf-file` is provided on the CLI, it overrides the checkpoint path inherited from `configs/longstream_infer.yaml` unless `--checkpoint` is also passed explicitly.

```bash
python run.py \
  --img-path /path/to/meta_root \
  --seq-list "seq list" \
  --hf-repo NicolasCC/LongStream \
  --hf-file 50_longstream.pt \
  --output-root outputs/seq00
```

## Quick Start and Inference Modes

Entrypoints:

- `run.py`: inference followed by evaluation
- `infer.py`: inference only
- `eval.py`: evaluation only on existing outputs

Inference modes:

- `batch_refresh`: cuts a long sequence into overlapping keyframe spans and stitches the outputs after duplicate removal.
- `streaming_refresh`: runs frame by frame with streaming caches and refreshes the caches at the same segment boundaries.

### Full Run With a Local Checkpoint

```bash
python run.py \
  --img-path /path/to/meta_root \
  --seq-list 00 \
  --checkpoint checkpoints/50_longstream.pt \
  --mode batch_refresh \
  --streaming-mode causal \
  --keyframe-stride 8 \
  --refresh 3 \
  --output-root outputs/
```

### Full Run Directly From Hugging Face

```bash
python run.py \
  --img-path /path/to/meta_root \
  --seq-list "seq list" \
  --hf-repo NicolasCC/LongStream \
  --hf-file 50_longstream.pt \
  --mode streaming_refresh \
  --streaming-mode causal \
  --window-size 48 \
  --output-root outputs/
```

### Inference Only

```bash
python infer.py \
  --img-path /path/to/meta_root \
  --seq-list 00 \
  --checkpoint checkpoints/50_longstream.pt \
  --output-root outputs/seq00
```

### Evaluation Only

```bash
python eval.py \
  --img-path /path/to/meta_root \
  --seq-list 00 \
  --output-root outputs/seq00
```

## Minimal Python API

```python
import yaml

from longstream.core.infer import run_inference_cfg

with open("configs/longstream_infer.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Point the loader to the prepared generalizable meta-root.
cfg["data"]["img_path"] = "/path/to/meta_root"

# Run only a subset of scenes. Remove this line to auto-discover scenes.
cfg["data"]["seq_list"] = ["00"]

# Use a local checkpoint.
cfg["model"]["checkpoint"] = "checkpoints/50_longstream.pt"

# Or, alternatively, load the checkpoint from Hugging Face.
# cfg["model"]["checkpoint"] = None
# cfg["model"]["hf"] = {
#     "repo_id": "NicolasCC/LongStream",
#     "filename": "50_longstream.pt",
# }

# Select the output directory for saved predictions.
cfg["output"]["root"] = "outputs/seq00_py"

# This writes poses, depths, point clouds, RGB/depth videos, and visualizations to disk.
run_inference_cfg(cfg)
```

`run_inference_cfg` saves predictions under `cfg["output"]["root"]`. The saved artifacts include per-frame poses and intrinsics, raw depth maps, colorized depth visualizations, RGB frame exports, and merged point clouds for both released branches.

## Important Runtime Args

| Arg | Meaning |
| --- | --- |
| `--mode` | `batch_refresh` or `streaming_refresh` |
| `--streaming-mode` | `causal` or `window` attention |
| `--window-size` | attention window for `window` mode |
| `--keyframe-stride` | keyframe interval |
| `--refresh` | keyframes per refresh span |
| `--camera` | select one camera in a multi-camera scene |
| `--mask-sky` | enable sky masking; requires `onnxruntime` and may auto-download `skyseg.onnx` |
| `--no-mask-sky` | disable sky masking, useful for offline runs or deterministic packaging |
| `--skip-eval` | skip evaluation in `run.py` |

## Demo

Install the demo requirements first:

To let the demos fetch the released checkpoint automatically from Hugging Face:

```bash
export LONGSTREAM_HF_REPO=NicolasCC/LongStream
export LONGSTREAM_HF_FILE=50_longstream.pt
```

Stable demo:

```bash
python demo_gradio.py
```

## Outputs

For each sequence, LongStream writes a dedicated directory under `output_root/<sequence>/`:

- `poses/abs_pose.txt`: per-frame absolute `w2c` extrinsics.
- `poses/rel_pose.txt`: per-frame relative pose-head outputs when the relative pose head is enabled.
- `poses/intri.txt`: per-frame intrinsics.
- `images/rgb/` and `images/rgb.mp4`: RGB frame exports and video.
- `depth/dpt/`: raw predicted depth maps as `.npy`.
- `depth/dpt_plasma/` and `depth/dpt_plasma.mp4`: colorized depth previews.
- `points/point_head*`: point clouds from the point-head branch.
- `points/dpt_unproj*`: point clouds obtained by unprojecting predicted depth with predicted poses.

When evaluation is run, metrics are saved to:

- `output_root/metrics/<sequence>.json`
- `output_root/summary.json`
- `output_root/plots/<sequence>_traj_3d.png`

## Citation

```bibtex
@article{cheng2026longstream,
  title={LongStream: Long-Sequence Streaming Autoregressive Visual Geometry},
  author={Chong Cheng and Xianda Chen and Tao Xie and Wei Yin and Weiqiang Ren and Qian Zhang and Xiaoyuang Guo and Hao Wang},
  journal={arXiv preprint arXiv:2602.13172},
  year={2026}
}
```

## Acknowledgements

We thank the authors and open-source projects of VGGT, Stream3R, StreamVGGT, DUSt3R, and CroCo for releasing code that informed this inference release.
