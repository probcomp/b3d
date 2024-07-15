from b3d.types import Array
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional
from b3d.io.video_input import VideoInput

@dataclass
class SegmentedVideoInput(VideoInput):
    """
    **Attributes (in addition to those inherited from VideoInput):**
    - segmentation
        segmented_video_input['segmentation'][t] is a int array of shape (H, W) containing the segmentation mask
        at time t.  The range of values is {-1, 0, 1, 2, ..., O - 1}, where O is the number of objects
        in the scene. The value -1 is used to indicate empty space in synthetic data.
        Shape: (T, H, W)
        Type: uint32
    - object_positions
        segmented_video_input['object_positions'] is a float32 array of shape (T, O, 3) containing the positions
        of the objects in the scene, at each time.
        Shape: (T, O, 3)
        Type: float32
    - object_quaternions
        segmented_video_input['object_quaternions'] is a float32 array of shape (T, O, 4) containing the quaternions
        of the objects in the scene, at each time.
        Shape: (T, O, 4)
        Type: float32
    - object_catalog_ids
        segmented_video_input['object_catalog_ids'] is a list of strings of length O, where O is the number of objects
        in the scene.  Each string is a name for the type of object that this object is an instance of.
        It is permitted for this field to be elided.
        Type: Optional[List[str]]
    """
    segmentation: Array
    object_positions: Array
    object_quaternions: Array
    object_catalog_ids: Optional[list] = None

    @property
    def oid(self):
        return self.segmentation

    def save(self, filepath):
        jnp.savez(filepath, **self.to_dict())
    
    @classmethod
    def load(cls, filepath: str):
        """Loads SegmentedVideoInput from file"""
        with open(filepath, 'rb') as f:
            data = jnp.load(f, allow_pickle=True)
            return cls(
                rgb=jnp.array(data['rgb']),
                xyz=jnp.array(data['xyz']),
                segmentation=jnp.array(data['segmentation']),
                camera_positions=jnp.array(data['camera_positions']),
                camera_quaternions=jnp.array(data['camera_quaternions']),
                camera_intrinsics_rgb=jnp.array(data['camera_intrinsics_rgb']),
                camera_intrinsics_depth=jnp.array(data['camera_intrinsics_depth']),
                object_positions=jnp.array(data['object_positions']),
                object_quaternions=jnp.array(data['object_quaternions']),
                object_catalog_ids=data['object_catalog_ids']
            )

    @classmethod
    def from_dict(cls, data):
        # TODO: Remove the `video::` prefix from the keys
        video_dict = {}
        for (k, v) in data.items():
            if k.startswith('video::'):
                video_dict[k[len('video::'):]] = v
                
        return cls(
            **video_dict,
            segmentation=data['segmentation'],
            object_positions=data['object_positions'],
            object_quaternions=data['object_quaternions'],
            object_catalog_ids=data['object_catalog_ids']
        )
    
    def to_dict(self):
        dict = {
            'segmentation': self.segmentation,
            'object_positions': self.object_positions,
            'object_quaternions': self.object_quaternions,
            'object_catalog_ids': self.object_catalog_ids
        }
        video_dict = super().to_dict()
        dict.update(video_dict)
        return dict

    def to_video_input(self):
        return VideoInput(
            self.xyz, self.rgb,
            self.camera_positions, self.camera_quaternions,
            self.camera_intrinsics_rgb, self.camera_intrinsics_depth
        )