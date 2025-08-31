import xml.etree.ElementTree as ET
import io

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


xml_content = """
<launch>
  <node pkg="tf2_ros" type="static_transform_publisher" name="camera_link_broadcaster"
      args="0.0333077 -0.0344275 -0.0403585   -0.0354012 -0.0346992 0.704407 0.708063 fr3_hand_tcp camera_color_optical_frame" />
</launch>
"""

xml_file = io.StringIO(xml_content)

tree = ET.parse(xml_file)
root = tree.getroot()

node = root.find(".//node[@name='camera_link_broadcaster']")
args_str = node.get('args')

args_list = args_str.split()[:7]

args_floats = [float(x) for x in args_list]

x, y, z = args_floats[0:3]
qx, qy, qz, qw = args_floats[3:7]

print(f"XYZ (Translation): ({x}, {y}, {z})")
print(f"Quaternion (Rotation): ({qx}, {qy}, {qz}, {qw})")

camera_pose = np.array([x, y, z, qw, qx, qy, qz])
transformation_matrix = pose_to_transformation_matrix(camera_pose)

print("Transformation Matrix:")
print(repr(transformation_matrix))
