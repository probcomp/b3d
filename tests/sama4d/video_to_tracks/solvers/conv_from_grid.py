import jax
import jax.numpy as jnp
import rerun as rr
import rerun.blueprint as rrb
from genjax import Pytree

from tests.common.solver import Solver

## Single patch tracker ##


@Pytree.dataclass
class SingleKeypointTracker2DState(Pytree):
    patch: jnp.ndarray
    pos2D: jnp.ndarray

    def __getitem__(self, key):
        return SingleKeypointTracker2DState(
            patch=self.patch[key], pos2D=self.pos2D[key]
        )


def initialize_tracker_state(frame: jnp.ndarray, pos2D: jnp.ndarray, patch_size: int):
    patch = get_patch_around_region_with_padding(frame, pos2D, size=patch_size)
    return SingleKeypointTracker2DState(patch=patch, pos2D=pos2D)


def update_tracker_state(
    state: SingleKeypointTracker2DState, frame: jnp.ndarray, reinitialize_patch: bool
):
    x, y = get_best_fit_pos(frame, state.patch)
    if reinitialize_patch:
        patch = get_patch_around_region_with_padding(
            frame, (x, y), size=state.patch.shape[0]
        )
    else:
        patch = state.patch
    return SingleKeypointTracker2DState(patch=patch, pos2D=jnp.array([x, y]))


## Multiple patch tracker (solver for VideoToTracksTask) ##


class KeypointTracker2DFromInitialGrid(Solver):
    """
    Overlays a grid of 2D points on the first frame, and attempts
    to track them through the video by using 2D convolution-based
    patch tracking.

    Options:
    - grid_size_x, grid_size_y: The number of points in the grid along
        the x and y axes.  (Will be evenly spaced across the frame.)
    - patch_size: The size of the square patch around each point to track.
    - do_reinitialization: Whether to reinitialize the patches at each frame.
    """

    grid_size_x: int
    grid_size_y: int
    patch_size: int
    do_reinitialization: bool

    tracker_states: list[SingleKeypointTracker2DState]  # batched

    def __init__(self, grid_size_x, grid_size_y, patch_size, do_reinitialization):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.patch_size = patch_size
        self.do_reinitialization = do_reinitialization
        self.tracker_states = []

    def get_initial_keypoints(self, height, width):
        x = jnp.linspace(0, width, self.grid_size_x)
        y = jnp.linspace(0, height, self.grid_size_y)
        grid = jnp.stack(jnp.meshgrid(x, y), axis=-1)
        return grid.reshape(-1, 2)

    def solve(self, task_specification, save_states=False):
        video = task_specification["video"]
        height, width, _ = video.shape[1:]
        keypoints = self.get_initial_keypoints(height, width)
        tracks = jnp.array([keypoints])
        tracker_states = jax.vmap(
            lambda pos2D: initialize_tracker_state(video[0], pos2D, self.patch_size)
        )(keypoints)
        if save_states:
            self.tracker_states.append(tracker_states)
        for frame in video[1:]:
            tracker_states = jax.vmap(
                lambda state: update_tracker_state(
                    state, frame, self.do_reinitialization
                )
            )(tracker_states)
            tracks = jnp.concatenate(
                [tracks, jnp.array([tracker_states.pos2D])], axis=0
            )
            if save_states:
                self.tracker_states.append(tracker_states)
        visibility = jnp.ones(tracks.shape[:2], dtype=bool)
        return {"keypoint_tracks": tracks, "keypoint_visibility": visibility}

    def visualize_solver_state(self, send_blueprint=True):
        if send_blueprint:
            rr.send_blueprint(self.rr_blueprint())

        for t in range(len(self.tracker_states)):
            rr.set_time_sequence("frame", t)
            for i in range(self.tracker_states[t].pos2D.shape[0]):
                state = self.tracker_states[t][i]
                rr.log(
                    f"/solver/patches/{i}",
                    rr.Image(state.patch),
                )
                rr.log(
                    f"/solver/boxes/{i}",
                    rr.Boxes2D(
                        mins=[
                            state.pos2D - self.patch_size // 2,
                            state.pos2D - self.patch_size // 2,
                        ],
                        sizes=[self.patch_size, self.patch_size],
                        class_ids=[i],
                    ),
                )

    def rr_blueprint(self):
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    contents=["task/video", "solver/boxes/**"]
                ),  # task video + boxes
                rrb.Tabs(
                    contents=[  # patches
                        rrb.Spatial2DView(contents=[f"solver/patches/{i}"])
                        for i in range(self.grid_size_x * self.grid_size_y)
                    ]
                ),
            )
        )


## Logic for finding the best fit position for a patch in a frame ##


def get_patch_around_region_with_padding(rgb, center, size, pad_value=-1):
    center = jnp.array(center, dtype=jnp.int32)
    x, y = center
    half_size = size // 2
    padded_rgb = jnp.pad(
        rgb,
        ((half_size, half_size), (half_size, half_size), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    return jax.lax.dynamic_slice(padded_rgb, (y, x, 0), (size, size, 3))


def patch_l1_error_at_position(rgb, center, patch):
    """Returns the L1 error between the patch and the patch centered at the given position."""
    return jnp.sum(
        jnp.abs(
            patch
            - get_patch_around_region_with_padding(rgb, center, size=patch.shape[0])
        )
    )


def get_errors_across_image(rgb, patch):
    height, width, _ = rgb.shape
    return jax.vmap(
        jax.vmap(
            lambda x, y: patch_l1_error_at_position(rgb, (x, y), patch),
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )(jnp.arange(0, height), jnp.arange(0, width))


def get_best_fit_pos(rgb, patch):
    errors = get_errors_across_image(rgb, patch)
    min_error = jnp.min(errors)
    y, x = jnp.where(errors == min_error, size=1)
    return x[0], y[0]
