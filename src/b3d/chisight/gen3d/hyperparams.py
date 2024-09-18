from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff, Pytree
from genjax import UpdateProblemBuilder as U
from jax.random import split
from tqdm import tqdm

import b3d
from b3d.chisight.gen3d.inference_moves import (
    get_pose_proposal_density,
    propose_other_latents_given_pose,
    propose_pose,
)
from b3d.chisight.gen3d.model import (
    dynamic_object_generative_model,
    get_hypers,
    get_new_state,
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)


@Pytree.dataclass
class InferenceHyperparams(Pytree):
    pass