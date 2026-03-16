import argparse
import os
import shutil
from glob import glob


def _read_list(path: str):
    with open(path, "r") as f:
        return [
            ln.strip()
            for ln in f.readlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]


def _write_list(path: str, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(f"{ln}\n")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str, copy: bool):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _link_tree(src_dir: str, dst_dir: str, copy: bool):
    _ensure_dir(dst_dir)
    files = sorted(glob(os.path.join(src_dir, "*")))
    for p in files:
        if os.path.isdir(p):
            continue
        _link_or_copy(p, os.path.join(dst_dir, os.path.basename(p)), copy=copy)


def _is_generalizable_meta_root(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "00", "images"))


def _is_kitti_sequences_root(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    if os.path.isdir(os.path.join(path, "sequences")):
        return True
    return os.path.isdir(os.path.join(path, "00")) and (
        os.path.isdir(os.path.join(path, "00", "image_2"))
        or os.path.isdir(os.path.join(path, "00", "image_02"))
    )


def _resolve_kitti_seq_dir(sequences_root: str, seq: str) -> str:
    if os.path.isdir(os.path.join(sequences_root, "sequences")):
        return os.path.join(sequences_root, "sequences", seq)
    return os.path.join(sequences_root, seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        help="KITTI source: generalizable meta_root OR KITTI odometry sequences root.",
    )
    parser.add_argument(
        "--out", required=True, help="Output meta_root in GeneralizableDataset format."
    )
    parser.add_argument(
        "--seqs",
        default=None,
        help="Comma-separated seq ids (e.g. 00,01,02). Default: use data_roots.txt or scan dirs.",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlinking."
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    copy = bool(args.copy)

    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    _ensure_dir(out)

    if _is_generalizable_meta_root(src):
        roots_file = os.path.join(src, "data_roots.txt")
        seqs = (
            _read_list(roots_file)
            if os.path.exists(roots_file)
            else sorted(
                [
                    d
                    for d in os.listdir(src)
                    if os.path.isdir(os.path.join(src, d, "images"))
                ]
            )
        )
        if args.seqs:
            seqs = [s for s in args.seqs.split(",") if s]
        for seq in seqs:
            src_scene = os.path.join(src, seq)
            dst_scene = os.path.join(out, seq)
            _ensure_dir(dst_scene)
            for sub in ["images", "cameras", "depths", "masks", "vis_depths"]:
                sp = os.path.join(src_scene, sub)
                if not os.path.exists(sp):
                    continue
                dp = os.path.join(dst_scene, sub)
                if os.path.isdir(sp):
                    if copy:
                        shutil.copytree(sp, dp, dirs_exist_ok=True)
                    else:
                        if not os.path.exists(dp):
                            os.symlink(sp, dp)
                else:
                    _link_or_copy(sp, dp, copy=copy)
        _write_list(os.path.join(out, "data_roots.txt"), seqs)
        return

    if not _is_kitti_sequences_root(src):
        raise RuntimeError("Unsupported KITTI source layout.")

    if args.seqs:
        seqs = [s for s in args.seqs.split(",") if s]
    else:
        sequences_root = (
            os.path.join(src, "sequences")
            if os.path.isdir(os.path.join(src, "sequences"))
            else src
        )
        seqs = sorted(
            [
                d
                for d in os.listdir(sequences_root)
                if os.path.isdir(os.path.join(sequences_root, d))
            ]
        )

    for seq in seqs:
        seq_dir = _resolve_kitti_seq_dir(src, seq)
        if not os.path.isdir(seq_dir):
            continue

        img2 = os.path.join(seq_dir, "image_2")
        if not os.path.isdir(img2):
            img2 = os.path.join(seq_dir, "image_02")
        img3 = os.path.join(seq_dir, "image_3")
        if not os.path.isdir(img3):
            img3 = os.path.join(seq_dir, "image_03")

        dst_scene = os.path.join(out, seq)
        dst_img2 = os.path.join(dst_scene, "images", "02")
        dst_img3 = os.path.join(dst_scene, "images", "03")
        if os.path.isdir(img2):
            _link_tree(img2, dst_img2, copy=copy)
        if os.path.isdir(img3):
            _link_tree(img3, dst_img3, copy=copy)

    _write_list(os.path.join(out, "data_roots.txt"), seqs)


if __name__ == "__main__":
    main()
