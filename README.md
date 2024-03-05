# JAX GL Renderer

## Installation
```
pip install git+https://github.com/probcomp/jax_gl_renderer.git
```

## API

#### render_to_barycentrics
```
renderer = JaxGLRenderer(image_width, image_height)
uvs, object_ids, triangle_ids  = renderer.render_to_barycentrics_many(
    poses, # Pose matrix with shape [num_scenes, num_objects, 4, 4] and dtype jnp.float32
    vertices, # Vertex position matrix with shape [num_vertices, 4] and dtype jnp.float32
    faces, # Triangle matrix with shape [num_triangles, 3] and dtype jnp.int32.
    ranges # Ranges matrix with shape [num_objects, 2] where the 2 elements specify start indices and counts into faces.
)

# `uvs` has shape [num_scenes, image_height, image_width, 2] and dtype jnp.float32
# `object_ids` has shape [num_scenes, image_height, image_width] and dtype jnp.int32
# `triangle_ids` has shape [num_scenes, image_height, image_width] and dtype jnp.int32
# Now consider scene number S and pixel at row i and column j.
# uvs[S, i, j] contains the `u` and `v` barycentric coordinates of the intersection point with triangle index at triangle_ids[S, i, j]
# which is on object index `object_ids[S, i, j] - 1`. If the pixel's ray did not intersect any triangle, the values in corresponding
# pixel in `uvs`, `objects_ids`, and `triangle_ids` will be 0.
```
`render_to_barycentrics` is the single scene version of `render_to_barycentrics_many`. So the input argument `poses` will be of size `[num_objects, 4, 4]` and the outputs `uvs`, `object_ids`, `triangle_ids` will be not have the first `num_scenes` dimension.

#### render_attribute
```
renderer = JaxGLRenderer(image_width, image_height)
interpolated_output  = renderer.render_attribute_many(
    poses, # Pose matrix with shape [num_scenes, num_objects, 4, 4] and dtype jnp.float32
    vertices, # Vertex position matrix with shape [num_vertices, 4] and dtype jnp.float32
    faces, # Triangle matrix with shape [num_triangles, 3] and dtype jnp.int32.
    ranges, # Ranges matrix with shape [num_objects, 2] where the 2 elements specify start indices and counts into faces.
    attributes # Attributes matrix with shape [num_vertices, num_attributes] which are the attributes corresponding to the vertices
)

# `interpolated_output` has shape [num_scenes, image_height, image_width, num_attributes] and dtype jnp.float32
# where the pixel value is the barycentric interpolation of the attributes corresponding to the 3 vertices of the triangle
# with which the pixel's ray intersected. If the pixel's ray does not intersect any triangle the value at that pixel will be 0s.
```
`render_attribute` is the single scene version of `render_attribute_many`. So the input argument `poses` will be of size `[num_objects, 4, 4]` and the output `interpolated_output` will be not have the first `num_scenes` dimension.

