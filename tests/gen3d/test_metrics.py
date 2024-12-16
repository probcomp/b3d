import b3d
import numpy as np
import pytest
from b3d.chisight.gen3d.metrics import (
    FP_RESULTS_ROOT_DIR,
    add_err,
    adds_err,
    compute_auc,
    foundation_pose_ycbv_result,
)

TEST_FP_YCBV_RESULT_DIR = (
    FP_RESULTS_ROOT_DIR / "ycbv/2024-07-11-every-50-frames-gt-init"
)
ycb_dir = b3d.get_assets_path() / "bop/ycbv/test"


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


@pytest.mark.parametrize("error_fn", [add_err, adds_err])
def test_ground_truth_pose(error_fn):
    pred_poses = gt_poses = np.random.rand(100, 4, 4)
    vertices = np.random.rand(100, 3)

    all_errors = []
    for pred_pose, gt_pose in zip(pred_poses, gt_poses):
        all_errors.append(error_fn(pred_pose, gt_pose, vertices))
    auc_score = compute_auc(all_errors)

    # error should be zero if pred_pose == gt_pose, and AUC should be 1 (highest)
    assert np.allclose(all_errors, 0.0)
    assert np.isclose(auc_score, 1.0)


@pytest.mark.skipif(
    not TEST_FP_YCBV_RESULT_DIR.exists() or not ycb_dir.exists(),
    reason="FoundationPose tracking result and YCBV dataset not found.",
)
def test_compute_metric():
    # example showing how the metric can be computed
    test_scene = 49
    obj_id = 0
    framerate = 50
    num_scenes = b3d.io.data_loader.get_ycbv_num_images(test_scene, subdir="test")
    image_ids = range(1, num_scenes + 1, framerate)
    all_data = b3d.io.data_loader.get_ycbv_data(test_scene, image_ids, subdir="test")

    # get the gt mesh
    obj_id_str = str(all_data[0]["object_types"][obj_id] + 1).rjust(6, "0")
    print(obj_id_str)
    mesh = b3d.Mesh.from_obj_file(ycb_dir / f"../models/obj_{obj_id_str}.ply").scale(
        0.001
    )

    # load the foundation pose tracking results
    fp_result = foundation_pose_ycbv_result.load(
        test_scene_id=test_scene, object_id=obj_id
    )

    # start computing the error
    all_adds_err = []
    for frame_data, pred_pose in zip(all_data, fp_result):
        camera_pose = frame_data["camera_pose"]
        obj_pose = frame_data["object_poses"][obj_id]
        gt_pose = (camera_pose.inv() @ obj_pose).as_matrix()
        all_adds_err.append(adds_err(pred_pose, gt_pose, mesh.vertices))

    # Note that in many of the paper, the results are aggregated per-object
    # (rather than per-scene)
    auc = compute_auc(all_adds_err)

    assert auc > 0.5
