import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import save_image, load_image
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera


# Define some global variables for vkitti2
vkitti2_img_pattern0 = 'frames/rgb/Camera_0/rgb_{frame:05d}.jpg'
vkitti2_img_pattern1 = 'frames/rgb/Camera_1/rgb_{frame:05d}.jpg'
vkitti2_dpt_pattern0 = 'frames/depth/Camera_0/depth_{frame:05d}.png'
vkitti2_dpt_pattern1 = 'frames/depth/Camera_1/depth_{frame:05d}.png'

# Define some global variables for EasyVolcap
easyvolcap_img_dir = 'images'
easyvolcap_dpt_dir = 'depths'
easyvolcap_cam_dir = 'cameras'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/original/vkitti2")
    parser.add_argument("--easyvolcap_root", default="data/vkitti2")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument('--scenes', nargs='+', default=[])
    args = parser.parse_args()

    # Global variables
    vkitti2_root = args.data_root
    easyvolcap_root = args.easyvolcap_root

    # Define per-scene preprocessing function
    def process_scene(scene):
        # Input and output paths
        src_root = join(vkitti2_root, scene)
        tar_root = join(easyvolcap_root, scene)

        # Load the raw camera parameters
        ixts = np.loadtxt(join(src_root, 'intrinsic.txt'), skiprows=1)
        exts = np.loadtxt(join(src_root, 'extrinsic.txt'), skiprows=1)

        # Check if there is any missing camera parameters or images or depth maps
        if not (
            len(ixts) // 2 == len(exts) // 2
            == len(os.listdir(join(src_root, dirname(vkitti2_img_pattern0))))
            == len(os.listdir(join(src_root, dirname(vkitti2_img_pattern1))))
            == len(os.listdir(join(src_root, dirname(vkitti2_dpt_pattern0))))
            == len(os.listdir(join(src_root, dirname(vkitti2_dpt_pattern1))))
        ):
            log(red(f"Missing camera parameters or images or depth maps in {cyan(scene)}"))
            return

        # Output camera parameters
        camera0 = dotdict()
        camera1 = dotdict()

        def process_view(sidx, tidx):
            # Load camera parameters
            K0, K1 = np.eye(3), np.eye(3)
            K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2] = ixts[tidx * 2 + 0][2:]
            K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2] = ixts[tidx * 2 + 1][2:]
            w2c0 = exts[tidx * 2 + 0][2:].reshape(4, 4)
            w2c1 = exts[tidx * 2 + 1][2:].reshape(4, 4)

            # Load depth maps
            dpt0 = cv2.imread(
                join(src_root, vkitti2_dpt_pattern0.format(frame=sidx)),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ) / 100.  # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
            dpt1 = cv2.imread(
                join(src_root, vkitti2_dpt_pattern1.format(frame=sidx)),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ) / 100.  # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/

            # Camera parameters
            camera0[f'{tidx:05d}'] = dotdict()
            camera1[f'{tidx:05d}'] = dotdict()
            cam0 = camera0[f'{tidx:05d}']
            cam1 = camera1[f'{tidx:05d}']
            cam0.H, cam0.W = dpt0.shape[0], dpt0.shape[1]
            cam1.H, cam1.W = dpt1.shape[0], dpt1.shape[1]
            cam0.K, cam0.R, cam0.T = K0, w2c0[:3, :3], w2c0[:3, 3:]
            cam1.K, cam1.R, cam1.T = K1, w2c1[:3, :3], w2c1[:3, 3:]
            cam0.D = np.zeros((1, 5), dtype=np.float32)
            cam1.D = np.zeros((1, 5), dtype=np.float32)

            # Link RGB image
            img0_path = join(tar_root, easyvolcap_img_dir, '00', f'{tidx:05d}.jpg')
            if not exists(img0_path):
                os.makedirs(dirname(img0_path), exist_ok=True)
                os.system(f"ln -sfn {abspath(join(src_root, vkitti2_img_pattern0.format(frame=sidx)))} {img0_path}")
            img1_path = join(tar_root, easyvolcap_img_dir, '01', f'{tidx:05d}.jpg')
            if not exists(img1_path):
                os.makedirs(dirname(img1_path), exist_ok=True)
                os.system(f"ln -sfn {abspath(join(src_root, vkitti2_img_pattern1.format(frame=sidx)))} {img1_path}")

            # Write the depth map
            dpt0_path = join(tar_root, easyvolcap_dpt_dir, '00', f'{tidx:05d}.exr')
            if not exists(dpt0_path):
                os.makedirs(dirname(dpt0_path), exist_ok=True)
                save_image(dpt0_path, dpt0.astype(np.float32))
            dpt1_path = join(tar_root, easyvolcap_dpt_dir, '01', f'{tidx:05d}.exr')
            if not exists(dpt1_path):
                os.makedirs(dirname(dpt1_path), exist_ok=True)
                save_image(dpt1_path, dpt1.astype(np.float32))


        # Find all views
        inds = [
            int(f[-9:-4]) for f in sorted(
                os.listdir(join(src_root, dirname(vkitti2_img_pattern0)))
            ) if f.endswith('.jpg')
        ]

        # Process all views parallelly
        parallel_execution(
            inds,
            list(range(len(inds))),
            action=process_view,
            sequential=False,
            num_workers=args.num_workers,
            print_progress=True,
        )

        # Write the camera data
        write_camera(camera0, join(tar_root, easyvolcap_cam_dir, '00'))
        write_camera(camera1, join(tar_root, easyvolcap_cam_dir, '01'))

        # Logging
        log(green(f"Processed scene {cyan(scene)}, total {len(inds):04d} views"))


    # Process all scenes
    if len(args.scenes) > 0:
        scenes = args.scenes
    else:
        # Get all scenes
        scenes = [
            f for f in sorted(os.listdir(vkitti2_root))
                if os.path.isdir(join(vkitti2_root, f))
        ]

    # Each scene of vkitti2 has some sub-scenes
    subscenes = []
    for scene in scenes:
        subscenes += [
            join(scene, f) for f in sorted(os.listdir(join(vkitti2_root, scene)))
                if os.path.isdir(join(vkitti2_root, scene, f))
        ]
    log(yellow(f"Found {len(subscenes)} sub-scenes in total"))

    # Process each scene
    for subscene in subscenes:
        process_scene(subscene)


if __name__ == "__main__":
    main()
