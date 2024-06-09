import jax.numpy as jnp
import b3d
from test.task_tests.tasks.triangle_posterior_identification import TrianglePosteriorIdentificationTask, DEFAULT_INTRINSICS
TPIT = TrianglePosteriorIdentificationTask

TASK_CONFIG = {
    "name": "monochrome-background--no-motion--oneframe",
    "task_args": {
        "scene_background": TPIT.get_monochrome_room_background(jnp.array([0., 0., 1.])),
        "foreground_triangle": {
            "color": jnp.array([1., 1., 1.]),
            "vertices": jnp.array([
                [5, 3, 7], [6, 1.5, 7], [6, 1.5, 8]   
            ], dtype=float)
        },
        "triangle_path": b3d.Pose.identity()[None, ...],
        "camera_path": b3d.camera_from_position_and_target(
            position=jnp.array([5., -0.5, 6.]),
            target=jnp.array([5, 3, 7.])
        )[None, ...]
    }
}

task = TPIT(**TASK_CONFIG["task_args"])
task.visualize_scene()