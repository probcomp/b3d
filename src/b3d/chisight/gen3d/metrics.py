from pathlib import Path

import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray

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

    def load(self, test_scene_id: IndentationError, object_id: int) -> FloatArray:
        """Given the test scene and object id, load the corresponding tracking
        result from the specified directory. The returning JAX array will have
        shape (num_frames, 4, 4), where the estimated pose in each frame is
        stored as a 4x4 transformation matrix.
        """
        filename = self.result_dir / str(test_scene_id) / f"object_{object_id}.npy"
        return jnp.load(filename)

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
