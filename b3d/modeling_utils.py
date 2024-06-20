import genjax
from b3d.pose import sample_uniform_pose, sample_gaussian_vmf_pose
from genjax.generative_functions.distributions import ExactDensity
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

class UniformDiscrete(ExactDensity, genjax.JAXGenerativeFunction):
    def sample(self, key, vals):
        return jax.random.choice(key, vals)

    def logpdf(self, sampled_val, vals, **kwargs):
        return jnp.log(1.0 / (vals.shape[0]))

class UniformPose(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, low, high):
        return sample_uniform_pose(key, low, high)

    def logpdf(self, pose, low, high):
        position = pose.pos
        valid = ((low <= position) & (position <= high))
        position_score = jnp.log((valid * 1.0) * (jnp.ones_like(position) / (high-low)))
        return position_score.sum() + jnp.pi**2

class VMF(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, mean, concentration):
        return tfp.distributions.VonMisesFisher(mean, concentration).sample(seed=key)

    def logpdf(self, x, mean, concentration):
        return tfp.distributions.VonMisesFisher(mean, concentration).log_prob(x)
vmf = VMF()

class GaussianPose(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, mean_pose, std, concentration):
        return sample_gaussian_vmf_pose(key, mean_pose, std, concentration)

    def logpdf(self, pose, mean_pose, std, concentration):
        translation_score = tfp.distributions.MultivariateNormalDiag(
        mean_pose.pos, jnp.ones(3) * std).log_prob(pose.pos)
        quaternion_score = tfp.distributions.VonMisesFisher(
            mean_pose.quat / jnp.linalg.norm(mean_pose.quat), concentration
        ).log_prob(pose.quat)
        return translation_score + quaternion_score

uniform_discrete = UniformDiscrete()
uniform_pose = UniformPose()
gaussian_vmf_pose = GaussianPose()
