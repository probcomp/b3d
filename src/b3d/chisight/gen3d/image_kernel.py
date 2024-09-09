from abc import abstractmethod

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import PRNGKey

import b3d


@Pytree.dataclass
class ImageLikelihood(genjax.ExactDensity):
    """An abstract class that defines the common interface for drift kernels."""

    @abstractmethod
    def sample(self, key: PRNGKey, new_state, hyperparams):
        raise NotImplementedError

    def logpdf(self, observed_rgbd, new_state, hyperparams):
        return self.info(observed_rgbd, new_state, hyperparams)["score"]

    def info_from_trace(self, trace):
        hyperparams, _ = trace.get_args()
        return self.info(
            trace.get_retval()["rgbd"], trace.get_retval()["new_state"], hyperparams
        )

    def info(self, observed_rgbd, new_state, hyperparams):
        raise NotImplementedError


@Pytree.dataclass
class SimpleNoRenderImageLikelihood(ImageLikelihood):
    def sample(self, key: PRNGKey, new_state, hyperparams):
        return jnp.zeros(
            (
                hyperparams["image_height"].const,
                hyperparams["image_width"].const,
                4,
            )
        )

    def logpdf(self, observed_rgbd, new_state, hyperparams):
        return self.info(observed_rgbd, new_state, hyperparams)["score"]

    def info(self, observed_rgbd, new_state, hyperparams):
        transformed_points = new_state["pose"].apply(hyperparams["vertices"])
        projected_pixel_coordinates = jnp.rint(
            b3d.xyz_to_pixel_coordinates(
                transformed_points,
                hyperparams["fx"],
                hyperparams["fy"],
                hyperparams["cx"],
                hyperparams["cy"],
            )
        ).astype(jnp.int32)

        observed_rgbd_masked = observed_rgbd[
            projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1]
        ]

        color_visible_branch_score = jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., :3], new_state["colors"], new_state["color_scale"]
        ).sum(axis=-1)
        color_not_visible_score = jnp.log(1 / 1.0**3)
        color_score = jnp.logaddexp(
            color_visible_branch_score + jnp.log(new_state["visibility_prob"]),
            color_not_visible_score + jnp.log(1 - new_state["visibility_prob"]),
        )

        depth_visible_branch_score = jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., 3],
            transformed_points[..., 2],
            new_state["depth_scale"],
        )
        depth_not_visible_score = jnp.log(1 / 1.0)
        _depth_score = jnp.logaddexp(
            depth_visible_branch_score + jnp.log(new_state["visibility_prob"]),
            depth_not_visible_score + jnp.log(1 - new_state["visibility_prob"]),
        )
        is_depth_non_return = observed_rgbd_masked[..., 3] < 0.0001

        non_return_probability = 0.05
        depth_score = jnp.where(
            is_depth_non_return, jnp.log(non_return_probability), _depth_score
        )

        lmbda = 0.5
        scores = lmbda * color_score + (1.0 - lmbda) * depth_score

        # Visualization
        latent_rgbd = jnp.zeros_like(observed_rgbd)
        latent_rgbd = latent_rgbd.at[
            projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], :3
        ].set(new_state["colors"])
        latent_rgbd = latent_rgbd.at[
            projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], 3
        ].set(transformed_points[..., 2])

        return {
            "score": scores.sum(),
            "scores": scores,
            "pixel_coordinates": projected_pixel_coordinates,
            "transformed_points": transformed_points,
            "observed_rgbd_masked": observed_rgbd_masked,
            "latent_rgbd": latent_rgbd,
        }
