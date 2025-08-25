import os
import numpy as np
import subprocess

import preproc.launch_hamer as hamer
from preproc.launch_slam import split_frames_shots, get_command, check_intrins
from preproc.extract_frames import split_frame

from loguru import logger


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def preprocess_frames(img_dir: str, src_path: str, overwrite: bool = False):
    """
    Extract frames from a video file.
    """
    if not overwrite and is_nonempty(img_dir):
        logger.info(
            f"Found {len(os.listdir(img_dir))} frames in {img_dir}, skipping extraction.")
        return
    logger.info(f"Extracting frames from {src_path} to {img_dir}.")

    out = split_frame(src_path, img_dir)
    if out < 1:
        raise ValueError("No frames extracted!")


def preprocess_tracks(datatype: str, img_dir: str, track_dir: str, shot_dir: str, gpu: int, overwrite: bool = False):
    """
    :param img_dir
    :param track_dir, expected format: res_root/track_name/sequence
    :param shot_dir, expected format: res_root/shot_name/sequence
    """
    if not overwrite and is_nonempty(track_dir):
        logger.info(f"Found tracks in {track_dir}, skipping preprocessing.")
        return

    logger.info(f"Running HaMeR on {img_dir}")
    track_root, seq = os.path.split(track_dir.rstrip("/"))
    res_root, track_name = os.path.split(track_root)
    shot_name = shot_dir.rstrip("/").split("/")[-2]

    hamer.process_seq(
        [gpu],
        res_root,
        seq,
        img_dir,
        track_name=track_name,
        shot_name=shot_name,
        datatype=datatype,
        overwrite=overwrite,
    )


def preprocess_cameras(cfg, overwrite: bool = False):
    if not overwrite and is_nonempty(cfg.sources.cameras):
        logger.info(
            f"Found cameras in {cfg.sources.cameras}, skipping preprocessing.")
        return

    logger.info(f"Running SLAM on {cfg.seq}")
    img_dir = cfg.sources.images
    map_dir = cfg.sources.cameras
    subseqs, shot_idcs = split_frames_shots(
        cfg.sources.images, cfg.sources.shots)
    logger.info(
        f"Shot indices: {shot_idcs}, Current shot index: {cfg.shot_idx}, Matched shot index: {np.where(shot_idcs == cfg.shot_idx)}")
    logger.info(f"Subsequences: {subseqs}")
    shot_idx = np.where(shot_idcs == cfg.shot_idx)[0][0]
    # run on selected shot
    start, end = subseqs[shot_idx]
    if not cfg.split_cameras:
        # only run on specified segment within shot
        end = start + cfg.end_idx
        start = start + cfg.start_idx
    intrins_path = cfg.sources.get("intrins", None)
    if intrins_path is not None:
        intrins_path = check_intrins(
            cfg.type, cfg.root, intrins_path, cfg.seq, cfg.split)

    print('img_dir, map_dir, start, end, intrins_path',
          img_dir, map_dir, start, end, intrins_path)

    cmd = get_command(
        img_dir,
        map_dir,
        start=start,
        end=end,
        intrins_path=intrins_path,
        overwrite=overwrite,
    )
    logger.info(f"Running command:\n{cmd}")
    gpu = cfg.gpu
    out = subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True)
    assert out == 0, "SLAM FAILED"
