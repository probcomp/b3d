from functools import partial
from pathlib import Path

import b3d
import fire
import numpy as np
import pandas as pd
from b3d.chisight.gen3d.metrics import (
    add_err,
    adds_err,
    compute_auc,
    foundation_pose_ycbv_result,
)
from b3d.io.data_loader import YCB_MODEL_NAMES
from tqdm.auto import tqdm

YCB_DIR = b3d.get_assets_path() / "bop/ycbv"

ALL_METRICS = {
    "ADD-S": adds_err,
    "ADD": add_err,
}

FRAME_RATE = 50


def collect_all_scores(get_pose_fn: callable):
    # e.g. all_score["ADD"]["002_master_chef_can"] gives the ADD error for the
    # object "002_master_chef_can"
    all_scores = {}
    for metric_name in ALL_METRICS:
        all_scores[metric_name] = {obj_name: [] for obj_name in YCB_MODEL_NAMES}

    # preload all gt meshes
    meshes = []
    for obj_id in range(len(YCB_MODEL_NAMES)):
        obj_id_str = str(obj_id + 1).rjust(6, "0")
        meshes.append(
            b3d.Mesh.from_obj_file(YCB_DIR / f"models/obj_{obj_id_str}.ply").scale(
                0.001
            )
        )

    for test_scene_id in range(48, 60):
        num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(YCB_DIR, test_scene_id)
        image_ids = range(1, num_scenes + 1, FRAME_RATE)
        print(f"Processing test scene {test_scene_id}")
        all_data = b3d.io.data_loader.get_ycbv_test_images(
            YCB_DIR, test_scene_id, image_ids
        )

        object_types = all_data[0]["object_types"]
        for idx, obj_id in tqdm(enumerate(object_types), desc="Processing objects"):
            obj_name = YCB_MODEL_NAMES[obj_id]
            obj_mesh = meshes[obj_id]
            pred_poses = get_pose_fn(test_scene_id, idx)
            # start computing the error for this object
            for frame_data, pred_pose in zip(all_data, pred_poses):
                # load ground truth pose
                camera_pose = frame_data["camera_pose"]
                obj_pose = frame_data["object_poses"][idx]
                gt_pose = (camera_pose.inv() @ obj_pose).as_matrix()
                # metrics
                for metric_name, metric_fn in ALL_METRICS.items():
                    all_scores[metric_name][obj_name].append(
                        metric_fn(pred_pose, gt_pose, obj_mesh.vertices)
                    )

    # aggregate results per object
    final_results = {}
    for metric_name in ALL_METRICS:
        final_results[metric_name] = {}
        for obj_name in YCB_MODEL_NAMES:
            final_results[metric_name][obj_name] = compute_auc(
                all_scores[metric_name][obj_name]
            )

    return pd.DataFrame(final_results), all_scores


def get_fp_pred_pose(test_scene_id: int, obj_id: int):
    return foundation_pose_ycbv_result.load(test_scene_id, obj_id)


def get_b3d_pred_pose(result_dir: Path, test_scene_id: int, obj_id: int):
    poses = np.load(
        result_dir / f"SCENE_{test_scene_id}_OBJECT_INDEX_{obj_id}_POSES.npy.npz",
    )
    poses = b3d.Pose(poses["position"], poses["quaternion"])
    return poses.as_matrix()


def main(b3d_result_dir: str, output_dir: str | None = None, get_fp_pose: bool = False):
    """
    Call this with `b3d_result_dir` as the directory containing `.npy.npz` files.
    """
    result_dir = Path(b3d_result_dir)
    if output_dir is None:
        output_dir = b3d_result_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_score_getter = partial(get_b3d_pred_pose, result_dir)

    if get_fp_pose:
        pred_score_getter = get_fp_pred_pose

    results_summary, _ = collect_all_scores(pred_score_getter)
    print(results_summary)
    if output_dir is not None:
        results_summary.to_csv(output_dir / "summary.csv")


if __name__ == "__main__":
    fire.Fire(main)
