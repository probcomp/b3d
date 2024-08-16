"""
The way this solver works is as follows.

We set a time interval I (e.g. I = 3) which is the number of frames
over which we have to stably track a keypoint to introduce it as a good keypoint.

At each time, we maintain:
1. A set of active keypoints `active_kp`
2. For each of j = 1, 2, ..., I, a set of possible keypoints to add,
    `possible_kp`.

The patch tracker attempts to maintain N total active keypoints.
It culls keypoints at each frame, and whenever it has M < N keypoints
due to culling, it adds keypoints from `possible_kp[I]` to get
up to N total keypoints (or as many as possible, if `possible_kp[I]`
is not a large enough set).

Each keypoint set is stored as a JAX data structure that can track up to
K tracks, but has some of these tracks labeled as inactive.

At time 1, we initialize a state where `possible_kp[1]` has a grid
of points (and there are no active keypoints).

At each time T > 1, we have an `old_state`, and we:
1. We drop `possible_kp[I]`
2. For each other keypoint set, we try extending each keypoint track.
    We then apply the culling criterion to inactivate some of them.
3. We then shift up each of the keypoint tracks.
4. We then instantiate a new `possible_kp[1]` at randomly chosen positions.
5. Now, we score the set `possible_kp[I]` against the active set, producing
    a score for each keypoint in `possible_kp[I]`.  (V1 does scoring just by
    considering which keypoints are far from existing keypoints.)
6. Add the top N-M scoring keypoints to the active set.

Possible improvements for a future iteration:
1. Rather than dropping `possible_kp[I]`, have some way of finding the best
    fits between the extensions of `possible_kp[I]` and `possible_kp[I-1]`,
    so our new `possible_kp[I]` is as good as possible.
2. Instantiate `possible_kp[1]` at non-randomly chosen positions.  Instead
    intelligently choose positions to fill in current gaps between the keypoints.
3. Cull keypoints from the active set as they get too close to each other, even
    if they are being tracked reasonably well.
4. Extend the method for scoring which keypoints to add, in a way that also involves
    something like the cumulitive error the tracks have, or how much
    the patch has changed.
5. Rather than adding the top N-M scoring keypoints to the active set, iteratively
    add the 1 top scoring keypoint to the active set, rescoring against the new active
    set at each step.
"""

from dataclasses import dataclass

import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

from tests.common.solver import Solver

from .conv_with_reinstantiation_utils import (
    get_best_fit_pos,
    get_patch_around_region_with_padding,
    replace_using_elements_from_other_vector,
)

### Params ###


@dataclass
class PatchTrackerParams:
    patch_size: int
    num_tracks: int
    frames_before_adding_to_active_set: int
    reinitialize_patches: bool
    culling_error_threshold: jnp.ndarray
    culling_error_ratio_threshold: jnp.ndarray
    mindist_for_second_error: jnp.ndarray
    maxdist_for_second_error: jnp.ndarray


### Single Track ###


@Pytree.dataclass
class KeypointTrack(Pytree):
    patch: jnp.ndarray
    pos2D: jnp.ndarray
    active: jnp.ndarray

    last_min_error: jnp.ndarray
    second_to_last_min_error: jnp.ndarray

    def __init__(self, patch, pos2D, active, last_min_error, second_to_last_min_error):
        self.patch = patch
        self.pos2D = jnp.array(pos2D, dtype=float)
        self.active = active
        self.last_min_error = last_min_error
        self.second_to_last_min_error = second_to_last_min_error

    @classmethod
    def init_empty(cls, patch_size):
        return cls(
            patch=jnp.zeros((patch_size, patch_size, 3)),
            pos2D=jnp.zeros(2),
            active=jnp.array(False),
            last_min_error=jnp.inf,
            second_to_last_min_error=jnp.inf,
        )

    @staticmethod  # (2, )
    def instantiate_at_pos(frame, pos2D: jnp.ndarray, patch_size):
        patch = get_patch_around_region_with_padding(frame, pos2D, size=patch_size)
        return KeypointTrack(
            patch,
            pos2D,
            active=True,
            last_min_error=jnp.inf,
            second_to_last_min_error=jnp.inf,
        )

    def update(self, frame, params: PatchTrackerParams):
        (x, y), min_error, second_min_error = get_best_fit_pos(
            frame,
            self.patch,
            params.mindist_for_second_error,
            params.maxdist_for_second_error,
        )

        if params.reinitialize_patches:
            patch = get_patch_around_region_with_padding(
                frame, (x, y), size=self.patch.shape[0]
            )
        else:
            patch = self.patch

        return KeypointTrack(
            patch, jnp.array([x, y]), self.active, min_error, second_min_error
        )

    def do_cull(self, params: PatchTrackerParams):
        cull_due_to_error = self.last_min_error > params.culling_error_threshold
        cull_due_to_ratio = (
            self.last_min_error / self.second_to_last_min_error
        ) > params.culling_error_ratio_threshold
        do_cull = cull_due_to_error | cull_due_to_ratio
        both_are_finite = jnp.isfinite(self.last_min_error) & jnp.isfinite(
            self.second_to_last_min_error
        )
        return both_are_finite & do_cull

    def cull(self, params: PatchTrackerParams):
        return KeypointTrack(
            patch=self.patch,
            pos2D=self.pos2D,
            active=self.active & (~self.do_cull(params)),
            last_min_error=self.last_min_error,
            second_to_last_min_error=self.second_to_last_min_error,
        )

    ## Visualization for debugging ##
    def rr_visualize_point(self, name):
        rr.log(
            f"{name}/positions",
            rr.Points2D(
                self.pos2D,
                colors=jax.vmap(
                    lambda is_active: jnp.where(
                        is_active, jnp.array([0.0, 1, 0]), jnp.array([1.0, 0, 0])
                    )
                )(self.active),
            ),
        )


### Track Batch ###


@Pytree.dataclass
class KeypointTrackSet(Pytree):
    batched_kpt: KeypointTrack

    @classmethod
    def init_empty(cls, patch_size, num_tracks):
        return KeypointTrackSet(
            jax.vmap(lambda _: KeypointTrack.init_empty(patch_size))(
                jnp.arange(num_tracks)
            )
        )

    def extend_all_keypoints(self, frame, params):
        return KeypointTrackSet(
            jax.vmap(lambda x: x.update(frame, params), in_axes=0)(self.batched_kpt)
        )

    def cull(self, params):
        return KeypointTrackSet(
            jax.vmap(lambda x: x.cull(params), in_axes=0)(self.batched_kpt)
        )

    @staticmethod
    def instantiate(key, frame, params):
        mins = jnp.array([params.patch_size / 2, params.patch_size / 2])
        maxs = jnp.array(frame.shape[:2], dtype=float) - mins
        points2d = (
            genjax.uniform.vmap(in_axes=(0, 0))
            .simulate(
                key,
                (
                    jnp.tile(mins, (params.num_tracks, 1)),
                    jnp.tile(maxs, (params.num_tracks, 1)),
                ),
            )
            .get_retval()
        )
        return KeypointTrackSet(
            jax.vmap(
                lambda pos2D: KeypointTrack.instantiate_at_pos(
                    frame, pos2D, params.patch_size
                )
            )(points2d)
        )

    def mindist_to_active_keypoint(self, pos):
        return jnp.min(
            jnp.where(
                self.batched_kpt.active,
                jnp.linalg.norm(self.batched_kpt.pos2D - pos, axis=1),
                jnp.inf,
            )
        )

    def add_tracks_from(self, other, params):
        # The logic here is:
        # 1. Compute the set of indices in the active set where we need to add keypoints
        # 2. Compute the min distance from each possible keypoint to each active keypoint in the active set
        # 3. Replace the keypoints in the active set with the top N-M scoring keypoints

        scores = jax.vmap(
            lambda keypoint: self.mindist_to_active_keypoint(keypoint.pos2D)
        )(other.batched_kpt)
        top_indices = jnp.argsort(
            jnp.where(other.batched_kpt.active, scores, jnp.inf), descending=True
        )

        # Replace the inactive keypoints with these top indices
        new_batched_kpt = jax.tree_map(
            lambda slf, othr: replace_using_elements_from_other_vector(
                self.batched_kpt.active, slf, othr[top_indices]
            ),
            self.batched_kpt,
            other.batched_kpt,
        )

        return KeypointTrackSet(new_batched_kpt)

    ## Visualization for debugging ##
    def rr_visualize_points(self, name):
        self.batched_kpt.rr_visualize_point(name)


### Full tracker state ###


@Pytree.dataclass
class TrackerState(Pytree):
    active_set: KeypointTrackSet
    possible_sets: list[KeypointTrackSet]

    @classmethod
    def init(cls, patch_size, num_tracks, frames_before_adding_to_active_set):
        return cls(
            active_set=KeypointTrackSet.init_empty(patch_size, num_tracks),
            possible_sets=[
                KeypointTrackSet.init_empty(patch_size, num_tracks)
                for _ in range(frames_before_adding_to_active_set)
            ],
        )

    def extend_all_keypoints(self, frame, params):
        return TrackerState(
            self.active_set.extend_all_keypoints(frame, params),
            [x.extend_all_keypoints(frame, params) for x in self.possible_sets],
        )

    def cull(self, params):
        return TrackerState(
            self.active_set.cull(params), [x.cull(params) for x in self.possible_sets]
        )

    def shift(self, params):
        return TrackerState(
            self.active_set,
            [KeypointTrackSet.init_empty(params.patch_size, params.num_tracks)]
            + self.possible_sets[:-1],
        )

    def instantiate_new(self, key, frame, params):
        return TrackerState(
            self.active_set,
            [self.possible_sets[0].instantiate(key, frame, params)]
            + self.possible_sets[1:],
        )

    def update_active_set(self, params):
        new_active_set = self.active_set.add_tracks_from(self.possible_sets[-1], params)
        # Eventual to-do: we could remove the values from `possible_sets[-1]` that were added
        # to the active set.
        # At the time I'm writing this comment, we will always drop this full set
        # at the next timestep anyway, so it doesn't matter.
        return TrackerState(new_active_set, self.possible_sets)

    ## Interface for the solver ##

    @classmethod
    def pre_init_state(cls, params):
        return cls.init(
            params.patch_size,
            params.num_tracks,
            params.frames_before_adding_to_active_set,
        )

    def update(self, key, frame, params):
        extended = self.extend_all_keypoints(frame, params)
        culled = extended.cull(params)
        shifted = culled.shift(params)
        reinstantiated = shifted.instantiate_new(key, frame, params)
        with_updated_active_set = reinstantiated.update_active_set(params)
        return with_updated_active_set

    def get_tracks_and_visibility(self):
        return (
            jax.vmap(lambda x: x.pos2D)(self.active_set.batched_kpt),
            jax.vmap(lambda x: x.active)(self.active_set.batched_kpt),
        )

    ## Visualization for debugging ##
    def rr_visualize(self):
        self.active_set.rr_visualize_points(name="active_set")
        for i, possible_set in enumerate(self.possible_sets):
            possible_set.rr_visualize_points(name=f"possible_set/{i}")


### Solver for VideoToTracksTask ###
class KeypointTracker2DWithReinitialization(Solver):
    params: PatchTrackerParams

    def __init__(self, **kwargs):
        self.params = PatchTrackerParams(**kwargs)

    def solve(self, task_specification):
        video = task_specification["video"]

        def step(state, key_and_frame):
            (key, frame) = key_and_frame
            new_state = state.update(key, frame, self.params)
            return (new_state, new_state.get_tracks_and_visibility())

        keys = jax.random.split(jax.random.PRNGKey(816527), video.shape[0])
        _, (keypoint_tracks, keypoint_visibility) = jax.lax.scan(
            step, TrackerState.pre_init_state(self.params), (keys, video)
        )

        return {
            "keypoint_tracks": keypoint_tracks,
            "keypoint_visibility": keypoint_visibility,
        }
