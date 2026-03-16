import argparse

from longstream.core.cli import (
    add_runtime_arguments,
    load_config_with_overrides,
    parse_runtime_args,
)
from longstream.core.infer import run_inference_cfg


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = parse_runtime_args(parser)
    cfg = load_config_with_overrides(args)
    run_inference_cfg(cfg)


if __name__ == "__main__":
    main()
