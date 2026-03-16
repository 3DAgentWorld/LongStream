import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional, Tuple

import torch

from longstream.utils.vendor.dust3r.utils.image import load_images_for_eval

dataset_metadata: Dict[str, Dict[str, Any]] = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,
    },
    "kitti": {
        "img_path": "data/kitti/sequences",
        "anno_path": "data/kitti/poses",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "image_2"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            anno_path, f"{seq}.txt"
        )
        if os.path.exists(os.path.join(anno_path, f"{seq}.txt"))
        else None,
        "traj_format": "kitti",
        "seq_list": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": None,
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "waymo": {
        "img_path": "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/scatt3r_evaluation/waymo_open_dataset_v1_4_3",
        "anno_path": None,
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "images",
            seq.split("_cam")[1] if "_cam" in seq else "00",
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "cameras",
            seq.split("_cam")[1] if "_cam" in seq else "00",
            "extri.yml",
        ),
        "traj_format": "waymo",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
}


@dataclass
class LongStreamSequenceInfo:
    name: str
    scene_root: str
    image_dir: str
    image_paths: List[str]
    camera: Optional[str]


class LongStreamSequence:
    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        image_paths: List[str],
        scene_root: Optional[str] = None,
        image_dir: Optional[str] = None,
        camera: Optional[str] = None,
    ):
        self.name = name
        self.images = images
        self.image_paths = image_paths
        self.scene_root = scene_root
        self.image_dir = image_dir
        self.camera = camera


def _read_list_file(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _is_generalizable_scene_root(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "images"))


def _direct_image_files(dir_path: str) -> List[str]:
    filelist = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpeg")))
    return filelist


class LongStreamDataLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.dataset = cfg.get("dataset", None)
        meta = dataset_metadata.get(self.dataset, {})
        self.img_path = cfg.get("img_path", meta.get("img_path"))
        self.mask_path = cfg.get("mask_path", meta.get("mask_path"))
        self.dir_path_func = meta.get("dir_path_func", lambda p, s: os.path.join(p, s))
        self.mask_path_seq_func = meta.get("mask_path_seq_func", lambda p, s: None)
        self.full_seq = bool(cfg.get("full_seq", meta.get("full_seq", True)))
        self.seq_list = cfg.get("seq_list", None)
        self.stride = int(cfg.get("stride", 1))
        self.max_frames = cfg.get("max_frames", None)
        self.size = int(cfg.get("size", 518))
        self.crop = bool(cfg.get("crop", False))
        self.patch_size = int(cfg.get("patch_size", 14))
        self.format = cfg.get("format", "auto")
        self.data_roots_file = cfg.get("data_roots_file", None)
        self.split = cfg.get("split", None)
        self.camera = cfg.get("camera", None)

    def _infer_format(self) -> str:
        if self.format in ["relpose", "generalizable"]:
            return self.format
        if self.img_path is None:
            return "relpose"
        if _is_generalizable_scene_root(self.img_path):
            return "generalizable"
        default_list = self.data_roots_file or "data_roots.txt"
        if os.path.exists(os.path.join(self.img_path, default_list)):
            return "generalizable"
        return "relpose"

    def _resolve_seq_list_generalizable(self) -> List[str]:
        if self.seq_list is not None:
            return list(self.seq_list)
        if self.img_path is None or not os.path.isdir(self.img_path):
            return []

        if _is_generalizable_scene_root(self.img_path):
            return [self.img_path]

        candidates = []
        if isinstance(self.data_roots_file, str) and self.data_roots_file:
            candidates.append(self.data_roots_file)
        if isinstance(self.split, str) and self.split:
            split_name = self.split.lower()
            if split_name in ["val", "valid", "validate"]:
                split_name = "validate"
            candidates.append(f"{split_name}_data_roots.txt")
        candidates.append("data_roots.txt")
        candidates.append("train_data_roots.txt")
        candidates.append("validate_data_roots.txt")

        for fname in candidates:
            path = os.path.join(self.img_path, fname)
            if os.path.exists(path):
                return _read_list_file(path)

        img_dirs = sorted(
            glob.glob(os.path.join(self.img_path, "**", "images"), recursive=True)
        )
        scene_roots = [os.path.dirname(p) for p in img_dirs]

        rels = []
        for p in scene_roots:
            try:
                rels.append(os.path.relpath(p, self.img_path))
            except ValueError:
                rels.append(p)
        return sorted(set(rels))

    def _resolve_seq_list_relpose(self) -> List[str]:
        if self.seq_list is not None:
            return list(self.seq_list)
        meta = dataset_metadata.get(self.dataset, {})
        if self.full_seq:
            if self.img_path is None or not os.path.isdir(self.img_path):
                return []
            seqs = [
                s
                for s in os.listdir(self.img_path)
                if os.path.isdir(os.path.join(self.img_path, s))
            ]
            return sorted(seqs)
        seqs = meta.get("seq_list", []) or []
        return list(seqs)

    def _resolve_seq_list(self) -> List[str]:
        fmt = self._infer_format()
        if fmt == "generalizable":
            return self._resolve_seq_list_generalizable()
        return self._resolve_seq_list_relpose()

    def _resolve_scene_root(self, seq_entry: str) -> Tuple[str, str]:
        if os.path.isabs(seq_entry) or os.path.sep in seq_entry:
            scene_root = seq_entry
            name = os.path.basename(os.path.normpath(seq_entry))
        else:
            scene_root = os.path.join(self.img_path, seq_entry)
            name = seq_entry
        return name, scene_root

    def _resolve_image_dir_generalizable(self, scene_root: str) -> Optional[str]:
        images_root = os.path.join(scene_root, "images")
        if not os.path.isdir(images_root):
            return None

        if isinstance(self.camera, str) and self.camera:
            cam_dir = os.path.join(images_root, self.camera)
            if os.path.isdir(cam_dir):
                return cam_dir

        if _direct_image_files(images_root):
            return images_root

        cams = [
            d
            for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        if not cams:
            return None
        cams = sorted(cams)

        frame_dirs = []
        for name in cams:
            child_dir = os.path.join(images_root, name)
            child_images = _direct_image_files(child_dir)
            if child_images:
                frame_dirs.append((name, len(child_images)))

        if (
            len(cams) > 10
            and len(frame_dirs) == len(cams)
            and max(count for _, count in frame_dirs) == 1
        ):
            return images_root

        return os.path.join(images_root, cams[0])

    def _camera_from_image_dir(self, image_dir: str) -> Optional[str]:
        parent = os.path.basename(os.path.dirname(image_dir))
        if parent != "images":
            return None
        return os.path.basename(image_dir)

    def _collect_filelist(self, dir_path: str) -> List[str]:
        filelist = _direct_image_files(dir_path)
        if not filelist:
            nested = []
            child_dirs = sorted(
                d for d in glob.glob(os.path.join(dir_path, "*")) if os.path.isdir(d)
            )
            for child_dir in child_dirs:
                child_images = _direct_image_files(child_dir)
                if child_images:
                    nested.append(child_images[0])
            filelist = nested
        if self.stride > 1:
            filelist = filelist[:: self.stride]
        if self.max_frames is not None:
            filelist = filelist[: self.max_frames]
        return filelist

    def _load_images(self, filelist: List[str]) -> torch.Tensor:
        views = load_images_for_eval(
            filelist,
            size=self.size,
            verbose=False,
            crop=self.crop,
            patch_size=self.patch_size,
        )
        imgs = torch.cat([view["img"] for view in views], dim=0)
        images = imgs.unsqueeze(0)
        images = (images + 1.0) / 2.0
        return images

    def iter_sequence_infos(self) -> Iterator[LongStreamSequenceInfo]:
        fmt = self._infer_format()
        seqs = self._resolve_seq_list()
        for seq_entry in seqs:
            if fmt == "generalizable":
                seq, scene_root = self._resolve_scene_root(seq_entry)
                dir_path = self._resolve_image_dir_generalizable(scene_root)
                if dir_path is None or not os.path.isdir(dir_path):
                    continue
                camera = self._camera_from_image_dir(dir_path)
            else:
                seq = seq_entry
                scene_root = os.path.join(self.img_path, seq)
                dir_path = self.dir_path_func(self.img_path, seq)
                if not os.path.isdir(dir_path):
                    continue
                camera = None

            filelist = self._collect_filelist(dir_path)
            if not filelist:
                continue
            yield LongStreamSequenceInfo(
                name=seq,
                scene_root=scene_root,
                image_dir=dir_path,
                image_paths=filelist,
                camera=camera,
            )

    def __iter__(self) -> Iterator[LongStreamSequence]:
        for info in self.iter_sequence_infos():
            print(
                f"[longstream] loading sequence {info.name}: {len(info.image_paths)} frames",
                flush=True,
            )
            images = self._load_images(info.image_paths)
            print(
                f"[longstream] loaded sequence {info.name}: {tuple(images.shape)}",
                flush=True,
            )
            yield LongStreamSequence(
                info.name,
                images,
                info.image_paths,
                scene_root=info.scene_root,
                image_dir=info.image_dir,
                camera=info.camera,
            )
