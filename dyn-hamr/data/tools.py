import os
import json
import functools
from typing import List

import numpy as np

from body_model import OP_NUM_JOINTS
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def read_keypoints(keypoint_fn: str):
    """
    Only reads body keypoint data of first person.
    """
    empty_kps = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
    if not os.path.isfile(keypoint_fn):
        return empty_kps

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data["people"]) == 0:
        print("WARNING: Found no keypoints in %s! Returning zeros!" % (keypoint_fn))
        return empty_kps

    person_data = data["people"][0]
    body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float32)
    body_keypoints = body_keypoints.reshape([-1, 3])
    return body_keypoints


def read_mask_path(path: str):
    mask_path = None
    if not os.path.isfile(path):
        return mask_path

    with open(path, "r") as f:
        data = json.load(f)

    person_data = data["people"][0]
    if "mask_path" in person_data:
        mask_path = person_data["mask_path"]

    return mask_path


def read_mano_preds(pred_path: str, tid: int, num_betas: int = 10):
    """
    reads the betas, body_pose, global orientation and translation of a mano prediction
    exported from phalp outputs
    returns betas (10,), body_pose (23, 3), global_orientation (3,), translation (3,)
    """
    pose = np.zeros((15, 3))
    rot = np.zeros(3)
    trans = np.zeros(3)
    betas = np.zeros(num_betas)
    if not os.path.isfile(pred_path):
        return pose, rot, trans, betas, int(tid)

    with open(pred_path, "r") as f:
        data = json.load(f)

    pose = np.array(data["body_pose"], dtype=np.float32)
    rot = np.array(data["global_orient"], dtype=np.float32)
    trans = np.array(data["cam_trans"], dtype=np.float32)
    betas = np.array(data["betas"], dtype=np.float32)
    is_right = np.array(data["is_right"], dtype=np.float32)

    return pose, rot, trans, betas, is_right


def load_mano_preds(
    pred_paths: List[str], tid: int, interp: bool = True, num_betas: int = 10
):
    vis_mask = np.array([os.path.isfile(x) for x in pred_paths])
    vis_idcs = np.where(vis_mask)[0]

    # load single image mano predictions
    stack_fnc = functools.partial(np.stack, axis=0)
    # (N, 16, 3), (N, 3), (N, 3), (N, 10)
    pose, orient, trans, betas, is_right = map(
        stack_fnc,
        zip(*[read_mano_preds(p, tid, num_betas=num_betas) for p in pred_paths]),
    )

    assert len(np.where(is_right != int(tid))[0]) == 0
    if not interp:
        return pose, orient, trans, betas, is_right

    # interpolate the occluded tracks
    orient_slerp = Slerp(vis_idcs, Rotation.from_rotvec(orient[vis_idcs]))
    trans_interp = interp1d(vis_idcs, trans[vis_idcs], axis=0)
    betas_interp = interp1d(vis_idcs, betas[vis_idcs], axis=0)

    tmin, tmax = min(vis_idcs), max(vis_idcs) + 1
    times = np.arange(tmin, tmax)
    orient[times] = orient_slerp(times).as_rotvec()
    trans[times] = trans_interp(times)
    betas[times] = betas_interp(times)

    # interpolate for each joint angle
    for i in range(pose.shape[1]):
        pose_slerp = Slerp(vis_idcs, Rotation.from_rotvec(pose[vis_idcs, i]))
        pose[times, i] = pose_slerp(times).as_rotvec()

    return pose, orient, trans, betas, is_right
