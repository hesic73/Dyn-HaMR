import os
import glob
import json
import numpy as np

import subprocess
from concurrent import futures
import multiprocessing as mp

from preproc.datasets import update_args


SRC_DIR = os.path.abspath(f"{__file__}/../")


def isimage(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext == ".png" or ext == ".jpg" or ext == ".jpeg"


def split_frames_equal(img_dir, seq_len=-1):
    """
    split the sequence into subsequences of seq_len
    returns start and end indices for each subsequence
    """
    if seq_len == -1:  # don't split
        return [(0, -1)]

    image_list = sorted(list(filter(isimage, os.listdir(img_dir))))
    num_imgs = len(image_list)
    splits = list(range(0, num_imgs, seq_len))
    splits[-1] = num_imgs  # add remainder to last subseq
    return list(zip(splits[:-1], splits[1:]))


def split_frames_shots(img_dir, shot_path, pad_shot=0, min_len=0):
    """
    split the sequence into subsequences according by shots in the video
    returns start and end indices for each subsequence
    """
    if shot_path is None or not os.path.isfile(shot_path):
        print(f"{shot_path} DOES NOT EXIST, USING WHOLE SEQUENCE")
        return [(0, -1)], [0]

    image_list = sorted(list(filter(isimage, os.listdir(img_dir))))#[:500]
    num_frames = len(image_list)

    with open(shot_path, "r") as f:
        shot_dict = json.load(f)

    # print(shot_dict.keys())
    shot_idcs = np.array([shot_dict[name] for name in image_list])
    subseqs = []
    for i in np.unique(shot_idcs):
        idcs = np.where(shot_idcs == i)[0]
        start, end = idcs.min(), idcs.max() + 1
        if pad_shot > 0:
            if start > 0:
                start = start + pad_shot
            if end < num_frames:
                end = end - pad_shot
            if (end - start) < min_len:  # this shot is too short
                continue
        subseqs.append((start, end))
    return subseqs, np.unique(shot_idcs)


def split_sequence(args, img_dir, seq):
    subseqs = split_frames_equal(img_dir, args.seq_len)
    return subseqs, [None for _ in subseqs]


def get_out_dir(args, seq, shot_idx=None, start=0, end=-1):
    if "egobody" in args.type or "3dpw" in args.type:
        if args.use_intrins:
            out_name = "cameras_intrins"
        else:
            out_name = "cameras_default"
    else:
        out_name = "cameras"

    if args.seq_len > 0:
        out_name = f"{out_name}_split"

    out_dir = f"{args.root}/slahmr/{args.split}/{out_name}/{seq}"

    if args.seq_len > 0:
        return f"{out_dir}/{start}-{end}"

    if args.type == "posetrack" and shot_idx is not None:
        return f"{out_dir}/shot-{shot_idx}"

    return out_dir


def get_intrins_path(data_type, data_root, seq):
    raise NotImplementedError


def get_command(img_dir, out_dir, start=0, end=-1, intrins_path=None, overwrite=False):
    cmd_args = [
        f"python {SRC_DIR}/run_slam.py",
        "-i",
        img_dir,
        "--map_dir",
        out_dir,
        "--disable_vis",
    ]
    if start != 0:
        cmd_args += ["--start", str(start)]
    if end != -1:
        cmd_args += ["--end", str(end)]

    if intrins_path is not None:
        cmd_args += ["--intrins_path", intrins_path]

    if overwrite:
        cmd_args.append("--overwrite")

    cmd = " ".join(cmd_args)
    return cmd

def get_slam_command(args, img_dir, seq, shot_idx=None, start=0, end=-1):
    out_dir = get_out_dir(args, seq, shot_idx, start, end)
    intrins_path = None
    if args.use_intrins:
        intrins_path = get_intrins_path(args.type, args.root, seq)
    return get_command(
        img_dir,
        out_dir,
        start=start,
        end=end,
        intrins_path=intrins_path,
        overwrite=args.overwrite,
    )


def launch_job(gpus, cmd):
    cur_proc = mp.current_process()
    print(cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1
    gpu = gpus[worker_id % len(gpus)]
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="posetrack", help="dataset to process")
    parser.add_argument(
        "--root", default=None, help="root of data to process, default None"
    )
    parser.add_argument(
        "--split", default="val", help="split of dataset to process, default val"
    )
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--seq_len", type=int, default=-1)
    parser.add_argument(
        "--use_intrins", action="store_true", help="use GT intrinsics if available"
    )
    parser.add_argument("-y", "--overwrite", action="store_true")
    args = parser.parse_args()
    args = update_args(args)

    print(f"Running SLAM on {len(args.seqs)} sequences")
    with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as ex:
        for img_dir, seq in zip(args.img_dirs, args.seqs):
            subseqs, shot_idcs = split_sequence(args, img_dir, seq)
            for shot, (start, end) in zip(shot_idcs, subseqs):
                cmd = get_slam_command(args, img_dir, seq, shot, start, end)
                print(cmd)
                ex.submit(launch_job, args.gpus, cmd)
