import jax
import b3d
import os
import unittest
import gc

def assert_valid_ftd(ftd):
    assert ftd.latent_keypoint_positions is not None
    assert ftd.observed_keypoints_positions is not None
    assert ftd.observed_features is not None
    assert ftd.rgbd_images is not None
    assert ftd.keypoint_visibility is not None
    assert ftd.object_assignments is not None
    assert ftd.camera_position is not None
    assert ftd.camera_quaternion is not None
    assert ftd.camera_intrinsics is not None

class FeatureTrackDataTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_unity_data(self):
        path = os.path.join(
        b3d.get_assets_path(),
            "shared_data_bucket/input_data/unity/keypoints/indoorplant/slidingBooks_60fps_lit_bg_800p.input.npz"
        )
        ftd = b3d.io.FeatureTrackData.load(path)
        assert_valid_ftd(ftd)

        ftd = ftd.slice_time(start_frame=21)
        assert_valid_ftd(ftd)

        ftd = ftd.slice_time(end_frame=21)
        assert_valid_ftd(ftd)

        ftd = ftd.slice_time(start_frame=21, end_frame=40)
        assert_valid_ftd(ftd)