# b3d - Bayes3D

## Installation

1. Install Pytorch:
```
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip3 install torch torchvision torchaudio
```
3. Install OpenGL:
```
sudo apt-get install mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```
3. Install JAX:
```
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
4. Install `b3d`:
```
pip install git+https://github.com/probcomp/b3d.git
```

### Install assets
1. Ask an admin to give your Google account 
    access to the Google cloud data bucket containing the assets.
    (Use the Google account associated with your Google cloud.)
2. Run `gcloud auth login` on your Google cloud machine,
    so the data-pulling script can use your Google account to
    authenticate pulling from the data bucket.
3. Run `b3d_pull` from within the conda environment
    in which you installed `b3d`.

## Renderer

Initialize `Renderer`:
```
image_width, image_height, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = jax_gl_renderer.Renderer(w, image_height, fx,fy, cx,cy, near, far)
```

#### render
```
uvs, object_ids, triangle_ids  = renderer.render_many(
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
- `zs`: has shape [num_scenes, image_height, image_width] and dtype jnp.float32

For scene number `S` and pixel at row `i` and column `j`, `uvs[S, i, j]` contains the `u` and `v` barycentric coordinates of the intersection point with triangle index at `triangle_ids[S, i, j]` which is on object index `object_ids[S, i, j] - 1`. And the z coordinate of the intersection point is `z[S, i, j]`. If the pixel's ray did not intersect any triangle, the values in corresponding.

#### render_attribute
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
