import ipywidgets as widgets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.gridspec import GridSpec

import b3d.chisight.gen3d.inference_moves as inference_moves


@jax.jit
def get_sample(
    key,
    observed_rgbd_for_point,
    previous_visibility_prob,
    previous_color,
    latent_depth,
    previous_dnrp,
    color_scale,
    depth_scale,
    hyperparams,
    inference_hyperparams,
):
    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    color_kernel = hyperparams["color_kernel"]
    obs_rgbd_kernel = hyperparams["image_kernel"].get_rgbd_vertex_kernel()
    rgb, visibility_prob, dnr_prob = inference_moves._propose_a_points_attributes(
        key,
        observed_rgbd_for_point,
        latent_depth,
        previous_color,
        previous_visibility_prob,
        previous_dnrp,
        depth_nonreturn_prob_kernel,
        visibility_prob_kernel,
        color_kernel,
        obs_rgbd_kernel,
        color_scale,
        depth_scale,
        inference_hyperparams,
        return_metadata=False,
    )[:3]
    return rgb, visibility_prob, dnr_prob


get_samples = jax.vmap(
    get_sample, in_axes=(0, None, None, None, None, None, None, None, None, None)
)


def plot_samples(samples, observed_rgbd_for_point, previous_color, latent_depth):
    fig = plt.figure(layout="constrained", figsize=(10, 10))
    gs = GridSpec(3, 3, figure=fig)

    fig.suptitle(f"Observed RGBD: {observed_rgbd_for_point}", fontsize=16)
    rgb, visibility_prob, dnr_prob = samples

    ax = fig.add_subplot(gs[0, 0])
    values, counts = jnp.unique(visibility_prob, return_counts=True)
    ax.bar(values, counts)
    ax.set_xticks(values)
    ax.set_title("Visibility Probability Samples")

    ax = fig.add_subplot(gs[0, 1])
    values, counts = jnp.unique(dnr_prob, return_counts=True)
    ax.bar(values, counts)
    ax.set_xticks(values)
    ax.set_title("Depth Nonreturn Probability Samples")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0.0, 2.0)
    ax.set_title("Depth")
    ax.axvline(
        x=observed_rgbd_for_point[3], color="black", linestyle="--", label="Observed"
    )
    ax.axvline(x=latent_depth, color="black", linestyle="dotted", label="Latent")
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(rgb[:, 0], jnp.linspace(0, 1, 100), color="r")
    ax.set_title("R Samples")
    ax.axvline(x=observed_rgbd_for_point[0], color="black", linestyle="--")
    ax.axvline(x=previous_color[0], color="black", linestyle="dotted")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(rgb[:, 1], jnp.linspace(0, 1, 100), color="g")
    ax.set_title("G Samples")
    ax.axvline(x=observed_rgbd_for_point[1], color="black", linestyle="--")
    ax.axvline(x=previous_color[1], color="black", linestyle="dotted")

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(rgb[:, 2], jnp.linspace(0, 1, 100), color="b")
    ax.set_title("B Samples")
    ax.axvline(
        x=observed_rgbd_for_point[2], color="black", linestyle="--", label="Observed"
    )
    ax.axvline(x=previous_color[2], color="black", linestyle="dotted", label="Previous")
    ax.legend()


def create_interactive_visualization(
    observed_rgbd_for_point, hyperparams, inference_hyperparams
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 1000)

    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    color_scale_kernel = hyperparams["color_scale_kernel"]
    depth_scale_kernel = hyperparams["depth_scale_kernel"]

    def f(
        previous_visibility_prob,
        previous_dnrp,
        latent_depth,
        previous_r,
        previous_g,
        previous_b,
        color_scale,
        depth_scale,
    ):
        previous_color = jnp.array([previous_r, previous_g, previous_b])
        previous_visibility_prob = float(previous_visibility_prob)
        previous_dnrp = float(previous_dnrp)
        samples = get_samples(
            keys,
            observed_rgbd_for_point,
            previous_visibility_prob,
            previous_color,
            latent_depth,
            previous_dnrp,
            color_scale,
            depth_scale,
            hyperparams,
            inference_hyperparams,
        )
        plot_samples(samples, observed_rgbd_for_point, previous_color, latent_depth)

    interact(
        f,
        previous_visibility_prob=widgets.ToggleButtons(
            options=[f"{x:.2f}" for x in visibility_prob_kernel.support],
            description="Prev Vis Prob:",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        ),
        previous_dnrp=widgets.ToggleButtons(
            options=[f"{x:.2f}" for x in depth_nonreturn_prob_kernel.support],
            description="Prev DNR Prob:",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        ),
        latent_depth=widgets.FloatSlider(
            value=observed_rgbd_for_point[3],
            min=-1.0,
            max=1.0,
            step=0.01,
            description="Latent Depth:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        previous_r=widgets.FloatSlider(
            value=observed_rgbd_for_point[0],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Previous R:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        previous_g=widgets.FloatSlider(
            value=observed_rgbd_for_point[1],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Previous G:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        previous_b=widgets.FloatSlider(
            value=observed_rgbd_for_point[2],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Previous B:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        color_scale=widgets.FloatSlider(
            value=color_scale_kernel.support.min(),
            min=color_scale_kernel.support.min(),
            max=color_scale_kernel.support.max(),
            step=0.001,
            description="Color Scale:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".4f",
        ),
        depth_scale=widgets.FloatSlider(
            value=depth_scale_kernel.support.min(),
            min=depth_scale_kernel.support.min(),
            max=depth_scale_kernel.support.max(),
            step=0.0005,
            description="Depth Scale:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".4f",
        ),
    )
