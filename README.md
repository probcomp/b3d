# jax_gl_renderer - Parallel OpenGL Rendering

## Installation

Install Pytorch.
```
sudo apt-get install mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
pip install git+https://github.com/probcomp/jax_gl_renderer.git
```

# Usage

Initialize `JaxGLRenderer`:
```
image_width, image_height, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = jax_gl_renderer.JaxGLRenderer(w, image_height, fx,fy, cx,cy, near, far)
```

### render_to_barycentrics
```
uvs, object_ids, triangle_ids  = renderer.render_to_barycentrics_many(
    poses, vertices, faces, ranges
)
```

Inputs:
-  `poses`: Pose matrix with shape [num_scenes, num_objects, 4, 4] and dtype jnp.float32
-  `vertices`: Vertex position matrix with shape [num_vertices, 4] and dtype jnp.float32
-  `faces`: Triangle matrix with shape [num_triangles, 3] and dtype jnp.int32
-  `ranges`: Ranges matrix with shape [num_objects, 2] where the 2 elements specify start indices and counts into faces.

Outputs:
- `uvs`: has shape [num_scenes, image_height, image_width, 2] and dtype jnp.float32
- `object_ids`: has shape [num_scenes, image_height, image_width] and dtype jnp.int32
- `triangle_ids`: has shape [num_scenes, image_height, image_width] and dtype jnp.int32

For scene number `S` and pixel at row `i` and column `j`, `uvs[S, i, j]` contains the `u` and `v` barycentric coordinates of the intersection point with triangle index at `triangle_ids[S, i, j]` which is on object index `object_ids[S, i, j] - 1`. If the pixel's ray did not intersect any triangle, the values in corresponding.

### render_attribute
```
image  = renderer.render_attribute_many(
    poses, vertices, faces, ranges, attributes
)
```
Inputs:
-  `poses`, `vertices`, `faces`, `ranges` are same as above.
-  `attributes`: Attributes matrix with shape [num_vertices, num_attributes] which are the attributes corresponding to the vertices

Outputs:
- `image` has shape [num_scenes, image_height, image_width, num_attributes] and dtype jnp.float32.  At each pixel the value is the barycentric interpolation of the attributes corresponding to the 3 vertices of the triangle with which the pixel's ray intersected. If the pixel's ray does not intersect any triangle the value at that pixel will be 0s.
