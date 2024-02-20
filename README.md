# JAX GL Renderer

Installation:
```
pip install git+https://github.com/probcomp/jax_gl_renderer.git
```

Usage:
```
renderer = JaxGLRenderer(image_width, image_height)
point_cloud_image = renderer.render(
    poses, # Pose tensor with shape [num_scenes, num_objects, 4, 4] and dtype jnp.float32
    vertices, # Vertex position tensor with shape [num_vertices, 4] and dtype jnp.float32
    faces, # Triangle tensor with shape [num_triangles, 3] and dtype jnp.int32.
    ranges # Ranges tensor with shape [num_objects, 2] where the 2 elements specify start indices and counts into tri.
)
# The output `point_cloud_image` has shape [num_scenes, image_height, image_width, 3] and dtype jnp.float32
# where the pixel value is the 3D coordinate of the intersection point (zeros if no intersection)
```
