import argparse
import os
import sys

import yaml


def default_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "longstream_infer.yaml",
    )


def add_runtime_arguments(parser):
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to longstream config yaml.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset hint. Generic format works without it.",
    )
    parser.add_argument("--img-path", default=None)
    parser.add_argument(
        "--seq-list",
        default=None,
        help="Comma-separated sequence names. Default: auto-detect all sequences.",
    )
    parser.add_argument("--format", default=None, help="generalizable")
    parser.add_argument("--data-roots-file", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-file", default=None)
    parser.add_argument(
        "--mode", default=None, help="batch_refresh | streaming_refresh"
    )
    parser.add_argument("--streaming-mode", default=None, help="causal | window")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--keyframe-stride", type=int, default=None)
    parser.add_argument(
        "--refresh",
        type=int,
        default=None,
        help="Number of keyframes per refresh span, inclusive of both ends and including the segment start keyframe.",
    )
    parser.add_argument(
        "--keyframes-per-batch",
        dest="keyframes_per_batch_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--depth-rel-delta-threshold", type=float, default=None)
    parser.add_argument("--point-f1-threshold", type=float, default=None)
    parser.add_argument("--eval-max-points", type=int, default=None)
    parser.add_argument("--eval-voxel-size", type=float, default=None)
    parser.add_argument("--max-full-pointcloud-points", type=int, default=None)
    parser.add_argument("--max-frame-pointcloud-points", type=int, default=None)
    parser.add_argument("--save-frame-points", action="store_true")
    parser.add_argument("--no-save-frame-points", action="store_true")
    parser.add_argument("--no-align-scale", action="store_true")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--no-mask-sky", action="store_true")
    return parser


def parse_runtime_args(parser):
    argv = [arg for arg in sys.argv[1:] if arg.strip()]
    return parser.parse_args(argv)


def load_config_with_overrides(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("model", {})

    if args.device is not None:
        cfg["device"] = args.device

    if args.output_root is not None:
        cfg.setdefault("output", {})
        cfg["output"]["root"] = args.output_root

    if args.dataset is not None:
        cfg.setdefault("data", {})
        cfg["data"]["dataset"] = args.dataset

    if args.img_path is not None:
        cfg.setdefault("data", {})
        cfg["data"]["img_path"] = args.img_path

    if args.seq_list is not None:
        seqs = [s.strip() for s in args.seq_list.split(",") if s.strip()]
        cfg.setdefault("data", {})
        cfg["data"]["seq_list"] = seqs

    if args.format is not None:
        cfg.setdefault("data", {})
        cfg["data"]["format"] = args.format

    if args.data_roots_file is not None:
        cfg.setdefault("data", {})
        cfg["data"]["data_roots_file"] = args.data_roots_file

    if args.camera is not None:
        cfg.setdefault("data", {})
        cfg["data"]["camera"] = args.camera

    if args.max_frames is not None:
        cfg.setdefault("data", {})
        cfg["data"]["max_frames"] = args.max_frames

    if args.checkpoint is not None:
        cfg.setdefault("model", {})
        cfg["model"]["checkpoint"] = args.checkpoint

    if args.hf_repo is not None or args.hf_file is not None:
        cfg.setdefault("model", {})
        cfg["model"].setdefault("hf", {})
        if args.hf_repo is not None:
            cfg["model"]["hf"]["repo_id"] = args.hf_repo
        if args.hf_file is not None:
            cfg["model"]["hf"]["filename"] = args.hf_file
        # CLI Hugging Face args should override a checkpoint path inherited
        # from the config, unless the user explicitly passed --checkpoint.
        if args.checkpoint is None:
            cfg["model"]["checkpoint"] = None

    if args.mode is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["mode"] = args.mode

    if args.streaming_mode is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["streaming_mode"] = args.streaming_mode

    if args.window_size is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["window_size"] = args.window_size
        cfg["model"].setdefault("longstream_cfg", {})
        cfg["model"]["longstream_cfg"]["window_size"] = args.window_size

    if args.keyframe_stride is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["keyframe_stride"] = args.keyframe_stride
        cfg["model"].setdefault("longstream_cfg", {})
        cfg["model"]["longstream_cfg"].setdefault("rel_pose_head_cfg", {})
        cfg["model"]["longstream_cfg"]["rel_pose_head_cfg"][
            "keyframe_stride"
        ] = args.keyframe_stride

    refresh = args.refresh
    if refresh is None and args.keyframes_per_batch_legacy is not None:
        refresh = args.keyframes_per_batch_legacy + 1
    if refresh is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["refresh"] = refresh

    if args.depth_rel_delta_threshold is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["depth_rel_delta_threshold"] = args.depth_rel_delta_threshold

    if args.point_f1_threshold is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["point_f1_threshold"] = args.point_f1_threshold

    if args.eval_max_points is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["point_eval_max_points"] = args.eval_max_points

    if args.eval_voxel_size is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["point_eval_voxel_size"] = args.eval_voxel_size

    if args.max_full_pointcloud_points is not None:
        cfg.setdefault("output", {})
        cfg["output"]["max_full_pointcloud_points"] = args.max_full_pointcloud_points

    if args.max_frame_pointcloud_points is not None:
        cfg.setdefault("output", {})
        cfg["output"]["max_frame_pointcloud_points"] = args.max_frame_pointcloud_points

    if args.save_frame_points:
        cfg.setdefault("output", {})
        cfg["output"]["save_frame_points"] = True
    if args.no_save_frame_points:
        cfg.setdefault("output", {})
        cfg["output"]["save_frame_points"] = False

    if args.no_align_scale:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["align_scale"] = False

    if args.mask_sky:
        cfg.setdefault("output", {})
        cfg["output"]["mask_sky"] = True
    if args.no_mask_sky:
        cfg.setdefault("output", {})
        cfg["output"]["mask_sky"] = False

    infer_cfg = cfg.setdefault("inference", {})
    if "refresh" not in infer_cfg and "keyframes_per_batch" in infer_cfg:
        infer_cfg["refresh"] = int(infer_cfg["keyframes_per_batch"]) + 1

    cfg.setdefault("data", {})
    cfg["data"]["format"] = "generalizable"
    return cfg
