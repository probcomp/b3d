import genjax
import jax
import jax.numpy as jnp
import b3d

@genjax.static_gen_fn
def chisight_sparse_model(num_keypoints_array):
    xyz_relative = genjax.uniform_pose(
        -100.0 *  jnp.ones((len(num_keypoints_array), 3)),
        100.0 * jnp.ones((len(num_keypoints_array), 3))
    ) @ "xyz_relative"

    cluster_assignments = genjax.categoriacl()

    object_positions_over_time
    object_quaternions_over_time

    camera_position_over_time
    camera_quaternion_over_time

    xyz_in_world_frame = object_poses[clusrt_assignments] @ xyz_relative
    xyz_in_camera_frame = camera_pose_over_time.inv() @ xyz_in_world_frame

    return xyz_in_camera_frame

# GETTER SETTER METHODS for the trace

# VISUALIZATION METHODS

def chisight_sparse_likelihood(xyz_in_camera_frame)
    return projection_to_2d_and_noise(xyz_in_camera_frame) @ "image"

def chisight_dense_likelihood(xyz_in_camera_frame):
    for i in range(len(xyz_in_camera_frame)):
        vertices = genjax.uniform(-1.0, 1.0, shape=(100, 3))
        faces = genjax.uniform(0, 100, shape=(100, 3))
        vertex_colors = genjax.uniform(0, 1, shape=(100, 3))
    
    mesh = Mesh(vertices, faces, vertex_colors)
    return dense_image_likelihood(mesh, xyz_in_camera_frame[i])


renderer = b3d.RenderOriginal(trace)


    return xyz_relative

key = jax.random.PRNGKey(10)
trace = chisight_sparse_model.simulate(key, (jnp.arange(100),))
trace["xyz_relative"]



# INFERENCE