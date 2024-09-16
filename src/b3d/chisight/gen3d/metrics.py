from pathlib import Path
from typing import Sequence

import numpy as np
from genjax import Pytree
from scipy import spatial

from b3d.utils import get_assets_path

FP_RESULTS_ROOT_DIR = (
    get_assets_path() / "shared_data_bucket/foundation_pose_tracking_results"
)
DEFAULT_FP_YCBV_RESULT_DIR = (
    FP_RESULTS_ROOT_DIR / "ycbv/2024-07-11-every-50-frames-gt-init"
)


@Pytree.dataclass
class YCBVTrackingResultLoader(Pytree):
    """A utility class that loads precomputed YCBV tracking results from the
    specified directory as commanded. Note that this dataclass itself does not
    keep a copy of the result
    """

    result_dir: Path

    def load(self, test_scene_id: IndentationError, object_id: int) -> np.ndarray:
        """Given the test scene and object id, load the corresponding tracking
        result from the specified directory. The returning JAX array will have
        shape (num_frames, 4, 4), where the estimated pose in each frame is
        stored as a 4x4 transformation matrix.
        """
        filename = self.result_dir / str(test_scene_id) / f"object_{object_id}.npy"
        return np.load(filename)

    def get_scene_ids(self) -> list[int]:
        return sorted(
            [int(test_scene.name) for test_scene in self.result_dir.iterdir()]
        )

    def get_object_ids(self, test_scene_id: int) -> list[int]:
        scene_dir = self.result_dir / str(test_scene_id)
        prefix_length = len("object_")
        return sorted(
            [int(object_id.stem[prefix_length:]) for object_id in scene_dir.iterdir()]
        )


# a default loader for the most recently computed foundation pose tracking results
foundation_pose_ycbv_result = YCBVTrackingResultLoader(DEFAULT_FP_YCBV_RESULT_DIR)


def apply_transform(pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ vertices.T + pose[:3, 3][:, None]).T


def add_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray) -> float:
    """Compute the Average Distance (ADD) error between the predicted pose and the
    ground truth pose, given the vertices of the object.

    References:
    - https://github.com/thodan/bop_toolkit/blob/59c5f486fe3a7886329d9fc908935e40d3bc0248/bop_toolkit_lib/pose_error.py#L210-L224
    - https://github.com/NVlabs/FoundationPose/blob/cd3ca4bc080529c53d5e5235212ca476d82bccf7/Utils.py#L232-L240
    - https://github.com/chensong1995/HybridPose/blob/106c86cddaa52765eb82f17bd00fdc72b98a02ca/lib/utils.py#L36-L49

    Args:
        pred_pose (np.ndarray): A 4x4 transformation matrix representing the predicted pose.
        gt_pose (np.ndarray): A 4x4 transformation matrix representing the ground truth pose.
        vertices (np.ndarray): The vertices of shape (num_vertices, 3) in the object frame,
            representing the 3D model of the object. Note that we should be using the vertices
            from the ground truth mesh file instead of the reconstructed point cloud.
    """
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)
    return np.linalg.norm(pred_locs - gt_locs, axis=-1).mean()


def adds_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray) -> float:
    """Compute the Average Closest Point Distance (ADD-S) error between the predicted pose and the
    ground truth pose, given the vertices of the object. ADD-S is an ambiguity-invariant pose
    error metric which takes care of both symmetric and non-symmetric objects

    References:
    - https://github.com/thodan/bop_toolkit/blob/59c5f486fe3a7886329d9fc908935e40d3bc0248/bop_toolkit_lib/pose_error.py#L227-L247
    - https://github.com/NVlabs/FoundationPose/blob/cd3ca4bc080529c53d5e5235212ca476d82bccf7/Utils.py#L242-L253
    - https://github.com/chensong1995/HybridPose/blob/106c86cddaa52765eb82f17bd00fdc72b98a02ca/lib/utils.py#L51-L68

    Args:
        pred_pose (np.ndarray): A 4x4 transformation matrix representing the predicted pose.
        gt_pose (np.ndarray): A 4x4 transformation matrix representing the ground truth pose.
        vertices (np.ndarray): The vertices of shape (num_vertices, 3) in the object frame,
            representing the 3D model of the object. Note that we should be using the vertices
            from the ground truth mesh file instead of the reconstructed point cloud.
    """
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pred_locs)
    nn_dists, _ = nn_index.query(gt_locs, k=1)

    return nn_dists.mean()


def compute_auc(errs: Sequence, max_val: float = 0.1, step=0.001):
    """Compute the Area Under the Curve (AUC) of the pose tracking errors at
    different thresholds.

    Reference:
    - https://github.com/NVlabs/FoundationPose/blob/cd3ca4bc080529c53d5e5235212ca476d82bccf7/Utils.py#L255-L266

    Args:
        errs (Sequence): An sequence of pose tracking errors.
        max_val (float, optional): The upper bound of the threshold. Defaults to 0.1.
        step (float, optional): The step between two threshold. Defaults to 0.001.
    """
    from sklearn import metrics

    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val * 1)
    return auc
