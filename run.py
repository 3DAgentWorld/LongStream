import argparse

from longstream.core.cli import (
    add_runtime_arguments,
    load_config_with_overrides,
    parse_runtime_args,
)
from longstream.core.infer import run_inference_cfg
from longstream.eval import evaluate_predictions_cfg


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--skip-eval", action="store_true")
    args = parse_runtime_args(parser)

    cfg = load_config_with_overrides(args)
    print("[longstream] run: inference", flush=True)
    run_inference_cfg(cfg)
    if not args.skip_eval:
        print("[longstream] run: evaluation", flush=True)
        evaluate_predictions_cfg(cfg)
        print("[longstream] run: done", flush=True)


if __name__ == "__main__":
    main()
