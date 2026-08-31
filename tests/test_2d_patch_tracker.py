import jax

import b3d
from b3d.chisight.patch_tracking_2d.patch_tracker import PatchTracker2D


def test_patch_tracker():
    path = (
        b3d.get_assets_path()
        / "shared_data_bucket/dynamic_SfM/feature_track_data/pan_around_blocks.npz"
    )
    ftd_og = b3d.io.FeatureTrackData.load(str(path)).slice_time(20, 40)
    video = ftd_og.rgbd[..., :3]

    tracker = PatchTracker2D(
        patch_size=11,
        num_tracks=80,
        frames_before_adding_to_active_set=7,
        reinitialize_patches=True,
        culling_error_threshold=60,
        culling_error_ratio_threshold=0.8,
        mindist_for_second_error=4,
        maxdist_for_second_error=40,
    )

    keypoint_tracks, keypoint_visibility = (
        tracker.run_and_get_tracks_separated_by_dimension(jax.random.PRNGKey(0), video)
    )

    assert len(keypoint_tracks.shape) == 3
    assert len(keypoint_visibility.shape) == 2
    assert keypoint_tracks.shape[0] == keypoint_visibility.shape[0]
    assert keypoint_tracks.shape[1] == keypoint_visibility.shape[1]
    assert keypoint_tracks.shape[2] == 2
