import ipywidgets as widgets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.gridspec import GridSpec


def plot_samples(
    samples,
    observed_rgbd_for_point,
    latent_rgbd_for_point,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    color_scale,
):
    fig = plt.figure(layout="constrained", figsize=(10, 10))
    gs = GridSpec(3, 3, figure=fig)

    fig.suptitle(f"Observed RGBD: {observed_rgbd_for_point}", fontsize=16)
    rgb = samples["colors"]
    visibility_prob = samples["visibility_prob"]
    dnr_prob = samples["depth_nonreturn_prob"]

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
    # ax.set_xlim(0.0, 2.0)
    ax.set_title("Depth")
    ax.axvline(
        x=observed_rgbd_for_point[3], color="black", linestyle="--", label="Observed"
    )
    ax.axvline(
        x=latent_rgbd_for_point[3], color="black", linestyle="dotted", label="Latent"
    )
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
    observed_rgbd_for_point,
    latent_rgbd_for_point,
    hyperparams,
    inference_hyperparams,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    attribute_proposal_function,
):
    key = jax.random.PRNGKey(0)

    color_scale_kernel = hyperparams["color_scale_kernel"]
    depth_scale_kernel = hyperparams["depth_scale_kernel"]

    def f(
        observed_r,
        observed_g,
        observed_b,
        observed_d,
        color_scale,
        depth_scale,
    ):
        _observed_rgbd_for_point = jnp.array(
            [observed_r, observed_g, observed_b, observed_d]
        )
        samples = jax.vmap(attribute_proposal_function, in_axes=(0, *(None,) * 9))(
            jax.random.split(key, 100),
            _observed_rgbd_for_point,
            latent_rgbd_for_point,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            color_scale,
            depth_scale,
            hyperparams,
            inference_hyperparams,
        )
        plot_samples(
            samples,
            _observed_rgbd_for_point,
            latent_rgbd_for_point,
            previous_color,
            previous_visibility_prob,
            previous_dnrp,
            color_scale,
        )

    interact(
        f,
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
        observed_r=widgets.FloatSlider(
            value=observed_rgbd_for_point[0],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Observed R:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        observed_g=widgets.FloatSlider(
            value=observed_rgbd_for_point[1],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Observed G:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        observed_b=widgets.FloatSlider(
            value=observed_rgbd_for_point[2],
            min=0.0,
            max=1.0,
            step=0.01,
            description="Observed B:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
        observed_d=widgets.FloatSlider(
            value=observed_rgbd_for_point[3],
            min=-1.0,
            max=hyperparams["intrinsics"]["far"],
            step=0.01,
            description="Observed Depth:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        ),
    )
