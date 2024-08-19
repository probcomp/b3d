"""
A patch tracking module based on 2D image patches.
This can be run across a video to produce a set of keypoint tracks
that come into and out of existance over time, covering the full video.

***Technical details***

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
from functools import partial

import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

from .utils import (
    get_best_fit_pos,
    get_patch_around_region_with_padding,
    replace_using_elements_from_other_vector,
)

### Params ###


@dataclass
class PatchTrackerParams:
    """
    Parameters for the patch tracker.

    patch_size: int (the patch will be `patch_size` x `patch_size`; Davison recommends 11x11)
    num_tracks: int (the number of tracks to try to maintain)
    frames_before_adding_to_active_set: int (the number of frames to ensure a patch appears trackable
        before it gets added to the set of active keypoint patches)
    reinitialize_patches: bool (whether to reinitialize patches at each frame
        [helpful for robustness to rotation, etc., but can lead to patch drift])
    culling_error_threshold: float (if the L1 error between the patch and the best fit
        is above this threshold, the patch is culled.  I have used the value 60.)
    culling_error_ratio_threshold: float (Another hyperparameter for culling.
        After finding the best fit patch position at a frame, we find the next best fit
        between `mindist_for_second_error` and `maxdist_for_second_error`
        pixels away from the best fit.  If the error at the best fit, divided by the error
        at the next best fit, is above this threshold, the patch is culled [since
        presumably this feature is not really a distinguishable one].  I've used 0.8.)
    mindist_for_second_error: float (see `culling_error_ratio_threshold`)
    maxdist_for_second_error: float (see `culling_error_ratio_threshold`)
    """

    patch_size: int
    num_tracks: int
    frames_before_adding_to_active_set: int
    reinitialize_patches: bool
    culling_error_threshold: jnp.ndarray
    culling_error_ratio_threshold: jnp.ndarray
    mindist_for_second_error: jnp.ndarray
    maxdist_for_second_error: jnp.ndarray

    def __hash__(self):
        return hash(
            (
                self.patch_size,
                self.num_tracks,
                self.frames_before_adding_to_active_set,
                self.reinitialize_patches,
                self.culling_error_threshold,
                self.culling_error_ratio_threshold,
                self.mindist_for_second_error,
                self.maxdist_for_second_error,
            )
        )


### Single Track ###


@Pytree.dataclass
class KeypointTrack(Pytree):
    """
    Internally used class; represents a single keypoint track
    (or, as a batched Jax object, a collection of keypoint tracks).
    """

    patch: jnp.ndarray
    pos2D: jnp.ndarray
    active: jnp.ndarray
    is_new: jnp.ndarray  # True if the keypoint was added in the last frame

    last_min_error: jnp.ndarray
    second_to_last_min_error: jnp.ndarray

    def __init__(
        self, patch, pos2D, active, is_new, last_min_error, second_to_last_min_error
    ):
        self.patch = patch
        self.pos2D = jnp.array(pos2D, dtype=float)
        self.active = active
        self.is_new = is_new
        self.last_min_error = last_min_error
        self.second_to_last_min_error = second_to_last_min_error

    @classmethod
    def init_empty(cls, patch_size):
        return cls(
            patch=jnp.zeros((patch_size, patch_size, 3)),
            pos2D=jnp.zeros(2),
            active=jnp.array(False),
            is_new=jnp.array(False),
            last_min_error=jnp.inf,
            second_to_last_min_error=jnp.inf,
        )

    @staticmethod  # (2, )
    def instantiate_at_pos(frame, pos2D: jnp.ndarray, patch_size):
        patch = get_patch_around_region_with_padding(frame, pos2D, size=patch_size)
        return KeypointTrack(
            patch,
            pos2D,
            active=jnp.array(True),
            is_new=jnp.array(True),
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
            patch,
            jnp.array([x, y]),
            self.active,
            jnp.array(False),
            min_error,
            second_min_error,
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
            is_new=self.is_new,
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
    """
    Internally used class; represents a set of keypoint tracks.
    """

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

        # Mark the new keypoints as new
        new_batched_kpt = KeypointTrack(
            patch=new_batched_kpt.patch,
            pos2D=new_batched_kpt.pos2D,
            active=new_batched_kpt.active,
            is_new=(~self.batched_kpt.active & new_batched_kpt.active),
            last_min_error=new_batched_kpt.last_min_error,
            second_to_last_min_error=new_batched_kpt.second_to_last_min_error,
        )

        return KeypointTrackSet(new_batched_kpt)

    ## Visualization for debugging ##
    def rr_visualize_points(self, name):
        self.batched_kpt.rr_visualize_point(name)


### Full tracker state ###


@Pytree.dataclass
class TrackerState(Pytree):
    """
    Internally used class; represents the full state of the patch tracker.
    """

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

    def get_tracks_visibility_and_is_new(self):
        return (
            jax.vmap(lambda x: x.pos2D)(self.active_set.batched_kpt),
            jax.vmap(lambda x: x.active)(self.active_set.batched_kpt),
            jax.vmap(lambda x: x.is_new)(self.active_set.batched_kpt),
        )

    ## Visualization for debugging ##
    def rr_visualize(self):
        self.active_set.rr_visualize_points(name="active_set")
        for i, possible_set in enumerate(self.possible_sets):
            possible_set.rr_visualize_points(name=f"possible_set/{i}")


### Tracker object ###


@partial(jax.jit, static_argnums=0)
def _update(params, key, frame, state):
    new_state = state.update(key, frame, params)
    keypoint_tracks, keypoint_visibility, is_new = (
        new_state.get_tracks_visibility_and_is_new()
    )

    return new_state, (keypoint_tracks, keypoint_visibility, is_new)


@partial(jax.jit, static_argnums=0)
def _jitted_scan(params, state0, key, frames):
    def step(state, key_and_frame):
        (key, frame) = key_and_frame
        new_state, (keypoint_tracks, keypoint_visibility, is_new) = _update(
            params, key, frame, state
        )
        return (new_state, (keypoint_tracks, keypoint_visibility, is_new))

    keys = jax.random.split(key, frames.shape[0])
    _, (keypoint_tracks, keypoint_visibility, is_new) = jax.lax.scan(
        step, state0, (keys, frames)
    )
    return keypoint_tracks, keypoint_visibility, is_new


@Pytree.dataclass
class PatchTracker2D(Pytree):
    """
    A keypoint tracking module based on convolving 2D image patches against each frame
    of a video.
    """

    params: PatchTrackerParams

    def __init__(self, **kwargs):
        self.params = PatchTrackerParams(**kwargs)

    def initialize(self):
        """
        Get an initial patch tracker state.  This is the state
        before any frames have been processed.
        """
        return TrackerState.pre_init_state(self.params)

    def update(self, key: jax.random.PRNGKey, frame: jnp.ndarray, state: TrackerState):
        """
        Given a PRNGKey, the current tracker state, and the first unprocessed frame
        of video `frame` (as an array of shape (H, W, 3)), update the tracker state
        and output the keypoint positions and visibility at this frame.

        Returns `new_tracker_state`, output_for_this_frame` where `output_for_this_state`
        is a three-tuple containing
        - `keypoint_tracks` (A 3D array of shape (N, 2) N is the number of keypoints,
            containing the 2D positions of the keypoints.)
        - `keypoint_visibility` (A 2D array of shape (N,) where N is the number of keypoints,
            containing whether each keypoint is visible.)
        - `is_new` (A 2D array of shape (N,) where N is the number of keypoints.
            `is_new[t, n]` is True if the keypoint  being tracked at position `n` is replaced
            with a totally separate keypoint track at this frame.)

        *Important note*: the keypoint positions and visibility values at index `n` _do not necessarily
        refer to the same keypoint track_ across frames.
        This is to allow keypoint tracks to be culled and replaced by a fresh keypoint
        without needing to continually expand the memory footprint.
        The way you can tell when a keypoint is replaced by a new keypoint is by checking
        the `is_new` array.  A single keypoint track will consist of all the keypoint positions
        and visibility values at position `n` during a contiguous temporal sequence in which
        `is_new[t, n] == False`.  Then, when a `t` is hit where `is_new[t, n] == True`,
        a new keypoint track begins, stored in the same position `n` in the output arrays.

        `expand_keypoint_tracks` can be used to convert from keypoint tracks in this format
        to a format where each index `n` refers to a unique keypoint track.
        """
        return _update(self.params, key, frame, state)

    def run_and_get_tracks_separated_by_dimension(self, key, video):
        """
        Run the patch tracker across the video.

        Args:
        - key (jax.random.PRNGKey): The random key to use for the tracker.
        - video (T, H, W, 3): The video to run the tracker on.

        Returns:
        - keypoint_tracks (T, N, 2): The 2D positions of the keypoints.
        - keypoint_visibility (T, N): Whether the keypoints are visible.
        Each index in dimension 1 in these arrays will refer to a unique keypoint track.
        """
        keypoint_tracks, keypoint_visibility, is_new = _jitted_scan(
            self.params, self.initialize(), key, video
        )
        return expand_keypoint_tracks(keypoint_tracks, keypoint_visibility, is_new)


def expand_keypoint_tracks(keypoint_tracks_2d, visibility, is_new):
    """
    Args:
    - keypoint_tracks_2d (T, N, 2): The 2D positions of the keypoints.
    - visibility (T, N): Whether the keypoints are visible.
    - is_new (T, N): Whether the keypoints are new.

    Returns:
    - keypoint_tracks_2d (T, M, 2): The 2D positions of the keypoints, so that each index
        in dimension 1 refers to a unique keypoint track.
    - visibility (T, M): Whether the keypoints are visible.
    """

    T, N, _ = keypoint_tracks_2d.shape

    new_tracks = []
    new_visibility = []

    for n in range(N):
        changepoints = jnp.concatenate(
            [jnp.array([0]), jnp.where(is_new[:, n])[0], jnp.array([T])]
        )
        idx_pairs = jnp.stack([changepoints[:-1], changepoints[1:]], axis=1)
        for low, high in idx_pairs:
            # get the track with zeros everywhere but between the changepoints
            track = jnp.zeros((T, 2)).at[low:high].set(keypoint_tracks_2d[low:high, n])
            new_tracks.append(track)

            # get the visibility with zeros everywhere but between the changepoints
            vis = jnp.zeros(T, dtype=bool).at[low:high].set(visibility[low:high, n])
            new_visibility.append(vis)

    return jnp.stack(new_tracks, axis=1), jnp.stack(new_visibility, axis=1)
