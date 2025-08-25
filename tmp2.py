import numpy as np
from transforms3d.quaternions import quat2mat


def pose_to_transformation_matrix(pose: np.ndarray) -> np.ndarray:
    x, y, z = pose[:3]
    quat = pose[3:]
    R = quat2mat(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


if __name__ == "__main__":
    translation = [0.389974, -0.583844, 0.438101]
    # The quaternion format for transforms3d is (qw, qx, qy, qz)
    quaternion = [0.538407, -0.842358, -0.0158126, 0.0173306]

    camera_pose = np.array(translation + quaternion)
    transformation_matrix = pose_to_transformation_matrix(camera_pose)

    print("Transformation Matrix:")
    print(repr(transformation_matrix))
