import jax
import jax.numpy as jnp

import b3d


def raycast_to_image_nondeterministic(key, intrinsics, vertices_in_camera_frame, K):
    """
    Returns an array of shape (H, W, K) containing K point indices, or -1 to indicate no point was registered.
    """
    N_pts = vertices_in_camera_frame.shape[0]

    projected_pixel_coordinates = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            vertices_in_camera_frame,
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
        )
        - 0.5
    ).astype(jnp.int32)
    permutation = jax.random.permutation(key, N_pts)
    shuffled_pixel_coordinates = projected_pixel_coordinates[permutation]
    # shuffled_pixel_coordinates = projected_pixel_coordinates # = jax.random.permutation(key, projected_pixel_coordinates)

    random_indices = jax.random.randint(
        key, (N_pts,), 0, K
    )  # (N_pts,) array of random indices from 0 to K-1
    registered_pixel_indices = -jnp.ones(
        (intrinsics["height"], intrinsics["width"], K), dtype=int
    )
    registered_pixel_indices = registered_pixel_indices.at[
        shuffled_pixel_coordinates[:, 0],
        shuffled_pixel_coordinates[:, 1],
        random_indices,
    ].set(permutation)  # jnp.arange(N_pts))

    return registered_pixel_indices
