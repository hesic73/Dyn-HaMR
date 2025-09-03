import time
from pathlib import Path

import numpy as np
import tyro
import viser
import viser.transforms as tf


def main(
    npz_path: Path = Path("world_poses.npz"),
    hand_axes_length: float = 0.08,
    hand_axes_radius: float = 0.01,
    cam_frustum_scale: float = 0.15,
    cam_axes_length: float = 0.1,
    cam_axes_radius: float = 0.008,
    fps: float = 30.0,
) -> None:
    """Visualize camera and hand poses from world_poses.npz using viser."""
    data = np.load(npz_path)
    cam_euler_world = data["cam_euler_world"]  # (T, 3)
    cam_t_world = data["cam_t_world"]  # (T, 3)
    hand_trans_world = data["hand_trans_world"]  # (2, T, 3)
    hand_orient_world_euler = data["hand_orient_world_euler"]  # (2, T, 3)
    T = cam_euler_world.shape[0]

    server = viser.ViserServer()

    # Playback controls
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=T - 1, step=1, initial_value=0
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=fps
        )

    # Add world frame
    server.scene.add_frame(
        "/world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.15,
        axes_radius=0.01,
    )

    # Add camera frustum and axes
    cam_frustum = server.scene.add_camera_frustum(
        "/camera/frustum",
        fov=np.deg2rad(60),  # Arbitrary FOV
        aspect=4/3,  # Arbitrary aspect
        scale=cam_frustum_scale,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )
    cam_axes = server.scene.add_frame(
        "/camera/axes",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=cam_axes_length,
        axes_radius=cam_axes_radius,
    )

    # Add hand frames
    hand_frames = []
    for i in range(2):
        hand_frames.append(
            server.scene.add_frame(
                f"/hand_{i}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=hand_axes_length,
                axes_radius=hand_axes_radius,
            )
        )

    def update_scene(t: int):
        # Camera
        cam_rot = tf.SO3.from_rpy_radians(cam_euler_world[t][0], cam_euler_world[t][1], cam_euler_world[t][2])
        cam_frustum.wxyz = cam_rot.wxyz
        cam_frustum.position = cam_t_world[t]
        cam_axes.wxyz = cam_rot.wxyz
        cam_axes.position = cam_t_world[t]
        # Hands
        for i in range(2):
            hand_rot = tf.SO3.from_rpy_radians(hand_orient_world_euler[i, t][0], hand_orient_world_euler[i, t][1], hand_orient_world_euler[i, t][2])
            hand_frames[i].wxyz = hand_rot.wxyz
            hand_frames[i].position = hand_trans_world[i, t]

    # GUI callbacks
    @gui_timestep.on_update
    def _(_):
        if not gui_playing.value:
            update_scene(gui_timestep.value)

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        if not gui_playing.value:
            update_scene(gui_timestep.value)

    # Initial scene
    update_scene(0)

    # Playback loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % T
            update_scene(gui_timestep.value)
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
