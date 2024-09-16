import pytest
from b3d.chisight.gen3d.metrics import (
    FP_RESULTS_ROOT_DIR,
    foundation_pose_ycbv_result,
)

TEST_FP_YCBV_RESULT_DIR = (
    FP_RESULTS_ROOT_DIR / "ycbv/2024-07-11-every-50-frames-gt-init"
)


@pytest.mark.skipif(
    not TEST_FP_YCBV_RESULT_DIR.exists(),
    reason="No foundation pose tracking results found.",
)
def test_loading_precomputed_ycbv_results():
    precomputed_poses = foundation_pose_ycbv_result.load(test_scene_id=48, object_id=1)
    assert precomputed_poses.shape == (45, 4, 4)

    test_scenes = foundation_pose_ycbv_result.get_scene_ids()
    assert len(test_scenes) == 12

    obj_ids_scene_48 = foundation_pose_ycbv_result.get_object_ids(test_scene_id=48)
    assert len(obj_ids_scene_48) == 5
