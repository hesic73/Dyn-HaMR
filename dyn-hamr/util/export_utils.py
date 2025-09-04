import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles


def export_world_poses(results_path: str, save_path: str):
    data = np.load(results_path)
    print(list(data.keys()))
    trans = data['trans']  # (2, T, 3)
    init_body_pose = data['init_body_pose']  # (2, T, 15, 3)
    latent_pose = data['latent_pose']  # (2, T, 15, 3)
    root_orient = data['root_orient']  # (2, T, 3)
    betas = data['betas']  # (2, 10)
    is_right = data['is_right']  # (2, T)
    pose_body = data['pose_body']  # (2, T, 15, 3)
    cam_R = data['cam_R']  # (2, T, 3, 3)
    cam_t = data['cam_t']  # (2, T, 3)
    intrins = data['intrins']  # (4)

    # Camera pose in world frame (if needed)
    cam_R_wc = cam_R[0]  # (T, 3, 3) world-to-cam
    cam_t_wc = cam_t[0]  # (T, 3)
    cam_R_world = np.transpose(cam_R_wc, (0, 2, 1))  # (T, 3, 3)
    cam_t_world = -np.einsum('tij,tj->ti', cam_R_world, cam_t_wc)  # (T, 3)

    # Hands are already in world frame
    hand_trans_world = trans  # (2, T, 3)
    hand_orient_world = axis_angle_to_matrix(
        torch.from_numpy(root_orient).to(torch.float32))  # (2, T, 3, 3)
    hand_orient_world_euler = matrix_to_euler_angles(
        hand_orient_world, 'XYZ').numpy()  # (2, T, 3)

    # Convert camera rotation matrices to Euler angles (XYZ convention)
    cam_euler_world = matrix_to_euler_angles(torch.from_numpy(cam_R_world).to(
        torch.float32), 'XYZ').numpy()  # (T, 3)

    # Save to npz (only Euler angles, not matrices)
    np.savez(save_path,
             cam_euler_world=cam_euler_world,
             cam_t_world=cam_t_world,
             hand_trans_world=hand_trans_world,
             hand_orient_world_euler=hand_orient_world_euler)
