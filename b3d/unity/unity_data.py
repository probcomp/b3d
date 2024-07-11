import dataclasses
from b3d.types import Array
import jax.numpy as jnp
from typing import Optional
import numpy as np
from FBExtractor import FBExtractor
from typing import Dict


@dataclasses.dataclass(kw_only=True)
class UnityData:
    """
    Unity data class. Note: Spatial units are measured in meters.
    Unity coordinate system is left-handed with x right, y up, and z forward. 
    For object_positions, quaternions, and catalog; dynamic objects are indexed first, then static objects.

    Args:
        rgb:                            (T, H, W, 3) Float Array
        depth:                          (T, H, W) Float Array
        segmentation:                   (T, H, W) Float Array
        camera_position:                (T, 3) Float Array
        camera_quaternion:              (T, 4) Float Array
        camera_intrinsics:              (8,) Float Array of camera intrinsics, see `camera.py`.
        object_positions:               (T, O, 3) Float Array
        object_quaternions:             (T, O, 4) Float Array
        object_catalog_ids:             (O, ) Float Array
        latent_keypoint_positions:      (T, N, 3) Float Array
        keypoint_visibility:            (T, N) Boolean Array OR None
        object_assignments:             (N,) Int Array
        Nframe:                         T int
        Nobjects:                       O int
        Nkeypoints:                     N int
    """

    rgb: Array
    depth: Array
    segmentation: Array
    camera_position: Array
    camera_quaternion: Array
    camera_intrinsics: Array
    object_positions: Array
    object_quaternions: Array
    object_catalog_ids: Optional[Array] = None
    latent_keypoint_positions: Array
    keypoint_visibility: Optional[Array] = None
    object_assignments: Array
    Nframe: int
    Nobjects: int
    Nkeypoints: int
    fps: Optional[float] = None
    file_info: Optional[Dict[str, str]] = None


    def save(self, filepath: str):
        """Saves input to file"""
        to_save = {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
        jnp.savez(filepath, **to_save)

    @classmethod
    def load(cls, filepath: str):
        """Loads input from file"""
        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=False)
            return cls(**{k: jnp.array(v) for k, v in data.items()})  # type: ignore
    
    def subsample_keypoints(self):
        # Create a mask where each point is at least true once across all frames
        mask = np.any(self.keypoint_visibility, axis=0)

        # Apply the mask to subsample the data
        subsampled_visibility = self.keypoint_visibility[:, mask]
        subsampled_position = self.latent_keypoint_positions[:, mask, :]
        subsampled_object_assignment = self.object_assignments[mask]

        # Update the data in the class with the subsampled data
        self.keypoint_visibility = subsampled_visibility
        self.latent_keypoint_positions = subsampled_position
        self.object_assignments = subsampled_object_assignment

    @classmethod
    def from_zip(cls, zip_path):
        # Create an instance of FBExtractor
        extractor = FBExtractor(zip_path)
        
        # Extract all data using the extractor
        camera_intrinsics = extractor.extract_camera_intrinsics()
        Nframe, Nobjects, Nkeypoints, samplingrate = extractor.extract_metadata()
        
        # rgb, depth, segmentation = extractor.extract_videos(Nframe, width, height, far)
        rgb = extractor.extract_rgb()
        depth = extractor.extract_depth()
        segmentation = extractor.extract_segmentation()

        camera_position, camera_quaternion = extractor.extract_camera_poses()
        object_positions, object_quaternions = extractor.extract_object_poses()
        object_catalog_ids = extractor.extract_object_catalog()
        object_assignments = extractor.extract_keypoints_object_assignment()
        keypoint_positions, keypoint_visibility = extractor.extract_keypoints()
        file_info = extractor.extract_file_info()

        extractor.close()

        instance = cls(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_position=camera_position,
            camera_quaternion=camera_quaternion,
            camera_intrinsics=camera_intrinsics,
            object_positions=object_positions,
            object_quaternions=object_quaternions,
            object_catalog_ids=object_catalog_ids,
            latent_keypoint_positions=keypoint_positions,
            keypoint_visibility=keypoint_visibility,
            object_assignments=object_assignments,
            Nframe=Nframe,
            Nobjects=Nobjects,
            Nkeypoints=Nkeypoints, 
            fps=samplingrate,
            file_info=file_info
        )

        # Only keep keypoints that are visible in at least one frame
        instance.subsample_keypoints()

        # Return an instance of UnityData
        return instance