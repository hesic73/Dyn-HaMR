import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Optional
import open3d as o3d

def get_camera_image(cam: str, resolution: tuple[int, int] = (848, 480), exposure: Optional[int] = None):
    """
    Captures color and depth images from a Realsense camera.
    Args:
        cam: The serial number of the camera.
        resolution: A tuple (width, height) for the desired resolution.
        exposure: Manual exposure value for the color sensor.
    Returns:
        A tuple containing the color and depth images.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(cam)

    config.enable_stream(
        rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
    config.enable_stream(
        rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, 30)

    profile = pipeline.start(config)

    # Print intrinsics for each stream with its name
    for s in profile.get_streams():
        video_stream = s.as_video_stream_profile()
        print(
            f"{video_stream.stream_name()} intrinsics: {video_stream.get_intrinsics()}")

    color_sensor = profile.get_device().query_sensors()[1]
    if exposure is not None:
        color_sensor.set_option(rs.option.exposure, exposure)
    else:
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")

    align_to = rs.stream.color
    align = rs.align(align_to)

    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    print(intrinsics)

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    pipeline.stop()
    return color_image, depth_image


def get_pointcloud_from_intrinsics(color: np.ndarray, depth: np.ndarray, intrinsics, depth_scale: float):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx6 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    depth = depth.astype(np.float32) * depth_scale

    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)

    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    color = color.astype(np.float32) / 255.0
    points = np.concatenate((points, color), axis=-1)  # (height, width, 6)

    return points


if __name__ == '__main__':
    cam_serial = "147122073100"

    # Example 1: Use a specific resolution (e.g., 848x480)
    color, depth = get_camera_image(cam_serial, resolution=(848, 480))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'color_{cam_serial}_848x480.png', color)
    cv2.imwrite(f'depth_{cam_serial}_848x480.png', depth)
    
    fx = 601.35
    fy = 601.285
    cx = 422.923
    cy = 241.747

    # 创建内参矩阵
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    pc = get_pointcloud_from_intrinsics(color, depth, K, 0.0010000000474974513)

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :, :3].reshape(-1, 3))
    o3d_pc.colors = o3d.utility.Vector3dVector(pc[:, :, 3:6].reshape(-1, 3))

    o3d.visualization.draw_geometries([o3d_pc])