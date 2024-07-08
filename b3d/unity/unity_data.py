import dataclasses
from hgps.types import Array
import jax.numpy as jnp
from typing import Optional
import numpy as np
import json


@dataclasses.dataclass(kw_only=True)
class UnityData:
    """
    Unity data class. Note: Spatial units are measured in meters.
    Unity coordinate system is left-handed with x right, y up, and z forward. 

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
        latent_keypoint_visibility:     (T, N) Boolean Array OR None
        object_assignments:             (N,) Int Array
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
    latent_keypoint_visibility: Optional[Array] = None
    object_assignments: Array


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
        
    @classmethod
    def from_json(cls, json_file_path: str) -> 'UnityVideoData':
        with open(json_file_path, 'r') as file:
            data = json.load(file)

            # Extract fixed variables
            
            width = int(data['camera']['imageResolution']['x'])
            height = int(data['camera']['imageResolution']['y'])

            # Convert camera intrinsics from Unity
            fx = data['camera']['cameraFocalLength'] * width / data['camera']['sensorSize']['x']
            fy = data['camera']['cameraFocalLength'] * height / data['camera']['sensorSize']['y']
            cx = width / 2 + width * data['camera']['lensShift']['x']
            cy = height / 2 + height * data['camera']['lensShift']['y']
            near = data['camera']['cameraNearClippingPlane']
            far = data['camera']['cameraFarClippingPlane']
            camera_intrinsics = np.array([width, height, fx, fy, cx, cy, near, far])

            # Get static variables
            frame_data = data['frameData']
            Nframe = len(frame_data)
            static_object_count = len(data['staticObjects']['staticObjectsPosition'])
            dynamic_object_count = len(data['frameData'][0]['dynamicObjectsPosition'])
            object_count = static_object_count + dynamic_object_count

            # Initialize arrays
            rgb = np.empty((Nframe, height, width, 3), dtype=float)
            depth = np.empty([Nframe, height, width], dtype=float)
            camera_position = np.empty((Nframe, 3), dtype=float)
            camera_rotation = np.empty((Nframe, 4), dtype=float)
            segmentation = np.empty([Nframe, height, width], dtype=jnp.uint32)
            object_positions = np.empty([Nframe, object_count, 3], dtype=jnp.float32)
            object_quaternions = np.empty([Nframe, object_count, 4], dtype=jnp.float32)
            object_catalog_ids = np.empty(object_count, dtype=str)

            # Get catalog of objects
            object_catalog_ids = data['staticObjects']['objectCatalogIDs']

            # Process each frame
            for f, frame in enumerate(frame_data):
                time = frame['frameID']

                # Fill RGB array and flip vertically
                R = np.array(frame['R']).reshape((height, width))
                G = np.array(frame['G']).reshape((height, width))
                B = np.array(frame['B']).reshape((height, width))
                rgb[time] = np.stack([R, G, B], axis=-1)[::-1]

                # depth values that are equal to 0 corresponds to no vertex in the json; we convert these values to the intrinsics.far value
                D = np.array(frame['D']).reshape((height, width))
                depth[time] = np.flipud(D)
                depth[depth == 0] = far

                seg = np.array(frame['objectID']).reshape((height, width))
                segmentation[time] = np.flipud(seg)

                # Get camera position and rotation
                camera_position[time] = np.array([frame['cameraPos'][axis] for axis in ['x', 'y', 'z']])
                camera_rotation[time] = np.array([frame['cameraRot'][axis] for axis in ['x', 'y', 'z', 'w']])

                # Get dynamic objects positions and visibility
                for i in range(dynamic_object_count):
                    object_positions[time][i] = np.array([frame['dynamicObjectsPosition'][i][axis] for axis in ['x', 'y', 'z']])
                    object_quaternions[time][i] = np.array([frame['dynamicObjectsRotation'][i][axis] for axis in ['x', 'y', 'z', 'w']])
                # Increment with static objects
                for i in range(static_object_count):
                    object_positions[time][i + dynamic_object_count] = np.array([data['staticObjects']['staticObjectsPosition'][i][axis] for axis in ['x', 'y', 'z']])
                    object_quaternions[time][i + dynamic_object_count] = np.array([data['staticObjects']['staticObjectsRotation'][i][axis] for axis in ['x', 'y', 'z', 'w']])

        # Create and return the UnityVideoData instance
        return cls(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_position=camera_position,
            camera_quaternion=camera_rotation,
            camera_intrinsics=camera_intrinsics,
            object_positions=object_positions,
            object_quaternions=object_quaternions,
            object_catalog_ids=object_catalog_ids
        )