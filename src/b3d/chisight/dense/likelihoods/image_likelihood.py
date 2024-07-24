import genjax
from genjax import Pytree


def make_image_likelihood(intermediate_func):
    @Pytree.dataclass
    class ImageLikelihood(genjax.ExactDensity):
        def sample(self, key, rendered_rgbd, likelihood_args):
            return rendered_rgbd

        def logpdf(self, observed_rgbd, rendered_rgbd, likelihood_args):
            results = intermediate_func(observed_rgbd, rendered_rgbd, likelihood_args)
            return results["score"]

    image_likelihood = ImageLikelihood()
    return image_likelihood
