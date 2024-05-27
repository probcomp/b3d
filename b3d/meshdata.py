import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass(kw_only=True)
class MeshData:
    """
    Mesh data class. Note: Spatial units are measured in meters.

    Args:
            vertices_positions:     (T, N, 3) Float Array
            vertices_normals:       (T, N, 3) Float Array
            vertices_uv:            (N, 2) Float Array; corresponds to the texture coordinates associated with the vertex. [0,0] represents the bottom-left corner of the texture, and [1,1] represents the top-rigtht. 
            triangles:              (M, 3) Int Array; vertices indexes
            triangles_materials:    (M, ) Int Array; material index
            materials:              (A, B, H, W, 3) Uint8 Array; there are B RGB textures for each material. For each material, the first texture is the color map texture, the second one (optional) is the normal map texture, and the third one (optional), the specular map texture. 
    """

    vertices_positions: any
    vertices_normals: any
    vertices_uv: any
    triangles: any
    triangles_materials: any
    materials: any

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