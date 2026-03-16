import os
import copy
import cv2
import numpy as np
import shutil
import urllib.request

try:
    import onnxruntime
except Exception:
    onnxruntime = None

SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
SKYSEG_THRESHOLD = 0.5
_WARNED_MESSAGES = set()


def _warn_once(message: str):
    if message in _WARNED_MESSAGES:
        return
    print(message, flush=True)
    _WARNED_MESSAGES.add(message)


def run_skyseg(session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    input_name = session.get_inputs()[0].name
    result_map = session.run(None, {input_name: x})[0]
    return result_map[0, 0]


def _normalize_skyseg_output(result_map):
    result_map = np.asarray(result_map, dtype=np.float32)
    if result_map.size == 0:
        return result_map
    finite = np.isfinite(result_map)
    if not np.any(finite):
        return np.zeros_like(result_map, dtype=np.float32)
    result_map = np.nan_to_num(result_map, nan=0.0, posinf=1.0, neginf=0.0)
    max_value = float(result_map.max())
    min_value = float(result_map.min())
    if min_value >= 0.0 and max_value > 1.5:
        result_map = result_map / 255.0
    return np.clip(result_map, 0.0, 1.0)


def sky_mask_filename(image_path):
    parent = os.path.basename(os.path.dirname(image_path))
    name = os.path.basename(image_path)
    if parent:
        return f"{parent}__{name}"
    return name


def segment_sky(image_path, session, mask_filename=None):
    image = cv2.imread(image_path)
    if image is None:
        return None
    result_map = run_skyseg(session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))
    result_map_original = _normalize_skyseg_output(result_map_original)
    output_mask = np.zeros(result_map_original.shape, dtype=np.uint8)
    output_mask[result_map_original < SKYSEG_THRESHOLD] = 255
    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    return output_mask


def compute_sky_mask(image_paths, model_path: str, target_dir: str = None):
    if onnxruntime is None:
        _warn_once(
            "[longstream] sky masking disabled: onnxruntime is not available. "
            "Install onnxruntime or run with --no-mask-sky."
        )
        return None
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        try:
            _warn_once(
                f"[longstream] skyseg.onnx not found at {model_path}; "
                "attempting automatic download."
            )
            with urllib.request.urlopen(SKYSEG_URL) as src, open(
                model_path, "wb"
            ) as dst:
                shutil.copyfileobj(src, dst)
        except Exception as exc:
            _warn_once(f"[longstream] failed to download skyseg.onnx: {exc}")
            _warn_once(
                "[longstream] sky masking disabled for this run. "
                "Place skyseg.onnx locally or run with --no-mask-sky."
            )
            return None
    if not os.path.exists(model_path):
        _warn_once(
            f"[longstream] sky masking disabled: skyseg.onnx is unavailable at {model_path}. "
            "Place the file locally or run with --no-mask-sky."
        )
        return None
    try:
        session = onnxruntime.InferenceSession(model_path)
    except Exception as exc:
        _warn_once(f"[longstream] failed to load skyseg.onnx: {exc}")
        _warn_once(
            "[longstream] sky masking disabled for this run. "
            "Check the model file or run with --no-mask-sky."
        )
        return None
    masks = []
    for image_path in image_paths:
        mask_filepath = None
        if target_dir is not None:
            name = sky_mask_filename(image_path)
            mask_filepath = os.path.join(target_dir, name)
            if os.path.exists(mask_filepath):
                sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            else:
                sky_mask = segment_sky(image_path, session, mask_filepath)
        else:
            sky_mask = segment_sky(image_path, session, None)
        masks.append(sky_mask)
    return masks
