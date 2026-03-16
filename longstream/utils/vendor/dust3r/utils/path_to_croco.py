import os.path as path
import sys

HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, "../../croco"))
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, "models")

if path.isdir(CROCO_MODELS_PATH):

    sys.path.insert(0, CROCO_REPO_PATH)
else:
    raise ImportError(
        f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )
