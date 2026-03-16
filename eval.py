import argparse

from longstream.core.cli import (
    add_runtime_arguments,
    load_config_with_overrides,
    parse_runtime_args,
)
from longstream.eval import evaluate_predictions_cfg


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    args = parse_runtime_args(parser)
    cfg = load_config_with_overrides(args)
    evaluate_predictions_cfg(cfg)


if __name__ == "__main__":
    main()
