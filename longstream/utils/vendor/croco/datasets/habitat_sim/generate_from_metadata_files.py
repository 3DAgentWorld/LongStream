"""
Script generating commandlines to generate image pairs from metadata files.
"""
import argparse
import glob
import os

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--prefix",
        default="",
        help="Commanline prefix, useful e.g. to setup environment.",
    )
    args = parser.parse_args()

    input_metadata_filenames = glob.iglob(
        f"{args.input_dir}/**/metadata.json", recursive=True
    )

    for metadata_filename in tqdm(input_metadata_filenames):
        output_dir = os.path.join(
            args.output_dir,
            os.path.relpath(os.path.dirname(metadata_filename), args.input_dir),
        )

        if os.path.exists(os.path.join(output_dir, "metadata.json")):
            continue
        commandline = f"{args.prefix}python datasets/habitat_sim/generate_from_metadata.py --metadata_filename={metadata_filename} --output_dir={output_dir}"
        print(commandline)
