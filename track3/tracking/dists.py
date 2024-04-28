import genjax
from genjax.generative_functions.distributions import ExactDensity
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

# def sample_gaussian_vmf_pose_2d(key, mean_pose, variance, concentration):
    
#     translation = tfp.distributions.MultivariateNormalDiag(
#         mean_pose[:2], jnp.ones(2) * variance
#     ).sample(seed=key)
#     key = jax.random.split(key, 1)[0]
#     rotation = tfp.distributions.VonMisesFisher(
#         mean_pose[2:], concentration
#     ).sample(seed=key)
#     pose = jnp.concatenate([translation, rotation])
#     assert pose.shape == (3,)
#     return pose

class GaussianVMF2D(ExactDensity,genjax.JAXGenerativeFunction):
    """
    Gaussian VMF distribution for a 2D pose [x, y, theta].
    Args:
        - key : PRNG key
        - mean_pose : jnp.array([x, y, theta])
        - variance : float (variance for translation, same in x and y directions)
        - concentration : float (concentration for rotation)
    """

    def _get_dists(self, mean_pose, variance, concentration):
        return (
            tfp.distributions.MultivariateNormalDiag(
                mean_pose[:2], jnp.ones(2) * variance
            ),
            tfp.distributions.VonMises(
                mean_pose[2], concentration
            )
        )

    def sample(self, key, mean_pose, variance, concentration):
        translation_dist, rotation_dist = self._get_dists(
            mean_pose, variance, concentration
        )
        translation = translation_dist.sample(seed=key)
        key = jax.random.split(key, 1)[0]
        rotation = rotation_dist.sample(seed=key)
        pose = jnp.concatenate([translation, rotation[None]])
        assert pose.shape == (3,)
        return pose

    def logpdf(self, pose, mean_pose, variance, concentration):
        translation_dist, rotation_dist = self._get_dists(
            mean_pose, variance, concentration
        )
        translation_score = translation_dist.log_prob(pose[:2])
        rotation_score = rotation_dist.log_prob(pose[2])
        return translation_score + rotation_score

gaussian_vmf_2d = GaussianVMF2D()