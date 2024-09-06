### IMPORTS ###


import b3d
import jax
import jax.numpy as jnp
from b3d import Pose
from genjax import ChoiceMapBuilder as C
from genjax import Pytree

b3d.reload(b3d.chisight.dynamic_object_model.dynamic_object_model)
b3d.reload(b3d.chisight.dynamic_object_model.dynamic_object_inference)


def make_trace_and_condition_values():
    key = jax.random.PRNGKey(0)
    model_vertices = jax.random.uniform(key, (1000, 3), minval=-0.5, maxval=0.5)

    fx, fy, cx, cy = 100.0, 100.0, 50.0, 50.0
    image_height, image_width = 100, 100

    hyperparams = {
        "vertices": model_vertices,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_height": Pytree.const(image_height),
        "image_width": Pytree.const(image_width),
        "max_pose_position_shift": 0.1,
        "color_shift_scale": 0.1,
        "color_outlier_probability_shift_scale": 0.1,
        "depth_outlier_probability_shift_scale": 0.1,
    }

    template_pose = Pose.identity()
    model_colors = jnp.zeros((len(model_vertices), 3))
    color_outlier_probabilities = jnp.ones(len(model_vertices)) * 0.01
    depth_outlier_probabilities = jnp.ones(len(model_vertices)) * 0.01

    previous_state = {
        "pose": template_pose,
        "colors": model_colors,
        "color_outlier_probabilities": color_outlier_probabilities,
        "depth_outlier_probabilities": depth_outlier_probabilities,
    }

    choicemap = C.n()

    key = jax.random.PRNGKey(0)
    trace, _ = (
        b3d.chisight.dynamic_object_model.dynamic_object_model.dynamic_object_generative_model.importance(
            key, choicemap, (hyperparams, previous_state)
        )
    )

    # Overwrite colors
    colors = trace.get_choices()("colors").c.v
    new_colors = jnp.ones_like(colors)
    assert not jnp.allclose(trace.get_choices()("colors").c.v, new_colors)
    chm = jax.vmap(lambda idx: C["colors", idx].set(new_colors[idx]))(
        jnp.arange(len(new_colors))
    )
    trace = trace.update(key, chm)[0]
    assert jnp.allclose(trace.get_choices()("colors").c.v, new_colors)

    # Overwite pose
    # pose = trace.get_choices()("pose").v
    new_pose = Pose.from_translation(jnp.array([0.1, 0.1, 0.1]))
    chm = C["pose"].set(new_pose)
    trace = trace.update(key, chm)[0]
    assert jnp.allclose(trace.get_choices()("pose").v.pos, new_pose.pos)
    assert jnp.allclose(trace.get_choices()("pose").v.quat, new_pose.quat)

    # TODO test choicemap creators

    # Test proposals
    b3d.reload(b3d.chisight.dynamic_object_model.dynamic_object_inference)
    from b3d.chisight.dynamic_object_model.dynamic_object_inference import (
        propose_color_and_color_outlier_probability,
        propose_depth_outlier_probability,
        propose_pose,
    )

    sampled_values, _ = propose_depth_outlier_probability(
        trace, key, jnp.linspace(0.0, 1.0, 128)
    )
    print(sampled_values.max(), sampled_values.mean())
    propose_pose(trace, key, 0.2, 200.0)
    (
        _,
        sampled_color_outlier_probabilities,
        _,
    ) = propose_color_and_color_outlier_probability(
        trace, key, jnp.array([0.0, 0.5, 1.0])
    )
    print(
        sampled_color_outlier_probabilities.max(),
        sampled_color_outlier_probabilities.mean(),
    )
