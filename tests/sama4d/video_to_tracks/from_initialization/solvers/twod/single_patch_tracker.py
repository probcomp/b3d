import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec

import b3d
from tests.common.solver import Solver


class SinglePatchTracker2D(Solver):
    def __init__(self):
        self.patch = None
        self.inferred_patches = None

    def solve(self, task_specification, log_to_self=False):
        kp0 = task_specification["initial_keypoint_positions_2D"]
        assert kp0.shape == (
            1,
            2,
        ), "Currently only single-keypoint-tracking is supported."
        kp0 = kp0[0]
        rgb = task_specification["video"][:, :, :, :3]
        patch = get_patch_around_region_with_padding(rgb[0], kp0, size=15)
        bestfit_positions_x, bestfit_positions_y = jax.vmap(
            lambda img: get_best_fit_pos(img, patch)
        )(rgb)
        stacked_xy = jnp.stack([bestfit_positions_x, bestfit_positions_y], axis=-1)
        result = stacked_xy[:, None, :]

        if log_to_self:
            self.patch = patch
            self.inferred_patches = jax.vmap(
                lambda t: get_patch_around_region_with_padding(
                    rgb[t], stacked_xy[t], size=patch.shape[0]
                )
            )(jnp.arange(rgb.shape[0]))

        return result

    def get_patches_over_time(self, rgb, points_xy):
        return jax.vmap(
            lambda t: get_patch_around_region_with_padding(
                rgb[t], points_xy[t], size=self.patch.shape[0]
            )
        )(jnp.arange(rgb.shape[0]))

    def export_mp4(
        self, task_specification, true_keypoint_positions, solution, file_prefix
    ):
        """
        Export an mp4 with the video, the true and inferred keypoint positions,
        the patch being tracked, and the image patch centered at the inferred
        and true keypoint position at each frame.

        Will be saved in "assets/test_results/" + file_prefix + "tracking"
        where {...} will include the solver name.
        """
        assert self.patch is not None and self.inferred_patches is not None, (
            "Must have run solver.solve(ts, log_to_self=True) before exporting mp4."
        )

        video = task_specification["video"]
        patches_at_true_positions = self.get_patches_over_time(
            video[..., :3], true_keypoint_positions[:, 0, :]
        )
        create_video_keypoint_mp4_with_patches(
            video[..., :3],
            true_keypoint_positions[:, 0, :],
            solution[:, 0, :],
            self.patch,
            self.inferred_patches,
            patches_at_true_positions,
            output_filename=f"{file_prefix}tracking.mp4",
        )


### Patch tracking logic ###


def get_patch_around_region_with_padding(rgb, center, size=11, pad_value=-1):
    center = jnp.array(center, dtype=jnp.int32)
    x, y = center
    half_size = size // 2
    padded_rgb = jnp.pad(
        rgb,
        ((half_size, half_size), (half_size, half_size), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    return jax.lax.dynamic_slice(padded_rgb, (y, x, 0), (size, size, 3))


def patch_l1_error_at_position(rgb, center, patch):
    """Returns the L1 error between the patch and the patch centered at the given position."""
    return jnp.sum(
        jnp.abs(patch - get_patch_around_region_with_padding(rgb, center, size=15))
    )


def get_errors_across_image(rgb, patch):
    height, width, _ = rgb.shape
    return jax.vmap(
        jax.vmap(
            lambda x, y: patch_l1_error_at_position(rgb, (x, y), patch),
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )(jnp.arange(0, height), jnp.arange(0, width))


def get_best_fit_pos(rgb, patch):
    errors = get_errors_across_image(rgb, patch)
    min_error = jnp.min(errors)
    y, x = jnp.where(errors == min_error, size=1)
    return x[0], y[0]


### Visualization logic ###


# With help from Claude
def create_video_keypoint_mp4(video, pos_xy, inferred_pos_xy, output_filename, fps=3):
    """
    Create an MP4 video from video frames with true and inferred keypoint positions.

    :param video: numpy array of shape (T, H, W, 3) representing the video
    :param pos_xy: numpy array of shape (T, 2) representing true keypoint positions
    :param inferred_pos_xy: numpy array of shape (T, 2) representing inferred keypoint positions
    :param output_filename: string, name of the output MP4 file
    :param fps: int, frames per second for the output video
    """

    output_filename = b3d.get_assets_path() / "test_results" / output_filename

    fig, ax = plt.subplots(figsize=(8, 6))

    def animate(i):
        ax.clear()

        # Display the video frame
        ax.imshow(video[i])

        # Plot the true keypoint position
        ax.plot(pos_xy[i, 0], pos_xy[i, 1], "go", markersize=12)

        # Plot the inferred keypoint position
        ax.plot(inferred_pos_xy[i, 0], inferred_pos_xy[i, 1], "bo", markersize=8)

        # Remove axis ticks
        # ax.set_xticks([])
        # ax.set_yticks([])

        # Add legend
        true_patch = mpatches.Patch(color="green", label="True Position")
        inferred_patch = mpatches.Patch(color="blue", label="Inferred Position")
        ax.legend(
            handles=[true_patch, inferred_patch],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
        )

        # Add frame number
        ax.text(
            0.02,
            0.98,
            f"Frame: {i + 1}/{len(video)}",
            transform=ax.transAxes,
            verticalalignment="top",
            color="white",
            fontweight="bold",
        )

    # Create the animation
    anim = FuncAnimation(
        fig, animate, frames=len(video), interval=1000 / fps, blit=False
    )

    # Set up the FFmpeg writer
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)

    # Save as MP4
    anim.save(output_filename, writer=writer)
    plt.close(fig)


# With help from Claude
def create_video_keypoint_mp4_with_patches(
    video,
    pos_xy,
    inferred_pos_xy,
    patch_to_track,
    tracked_patches,
    patches_at_gt_points,
    output_filename,
    fps=3,
):
    """
    Create an MP4 video from video frames with true and inferred keypoint positions,
    along with two PxP image patches.

    :param video: numpy array of shape (T, H, W, 3) representing the video
    :param pos_xy: numpy array of shape (T, 2) representing true keypoint positions
    :param inferred_pos_xy: numpy array of shape (T, 2) representing inferred keypoint positions
    :param patch_to_track: numpy array of shape (P, P, 3) representing the patch being tracked
    :param tracked_patches: numpy array of shape (T, P, P, 3) representing the patch tracked at each frame
    :param output_filename: string, name of the output MP4 file
    :param fps: int, frames per second for the output video
    """
    output_filename = b3d.get_assets_path() / "test_results" / output_filename

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, width_ratios=[3, 1, 1], height_ratios=[1, 1])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_patch1 = fig.add_subplot(gs[0, 1])
    ax_patch2 = fig.add_subplot(gs[1, 1])
    ax_patch3 = fig.add_subplot(gs[1, 2])

    P = patch_to_track.shape[0]
    fig.suptitle(f"Patch tracking with static {P}x{P} 2D patches", fontsize=16)

    def animate(i):
        # Clear all axes
        ax_main.clear()
        ax_patch1.clear()
        ax_patch2.clear()

        # Display the main video frame
        ax_main.imshow(video[i])
        ax_main.plot(pos_xy[i, 0], pos_xy[i, 1], "go", markersize=12)
        ax_main.plot(inferred_pos_xy[i, 0], inferred_pos_xy[i, 1], "bo", markersize=8)
        # ax_main.set_xticks([])
        # ax_main.set_yticks([])

        # Add frame number
        ax_main.text(
            0.02,
            0.98,
            f"Frame: {i + 1}/{len(video)}",
            transform=ax_main.transAxes,
            verticalalignment="top",
            color="white",
            fontweight="bold",
        )

        # Display the patch being tracked
        ax_patch1.imshow(patch_to_track)
        ax_patch1.set_title("Patch being tracked")
        # ax_patch1.set_xticks([])
        # ax_patch1.set_yticks([])

        ax_patch2.imshow(tracked_patches[i])
        ax_patch2.set_title("Image at current\ninferred keypoint position")
        # ax_patch2.set_xticks([])
        # ax_patch2.set_yticks([])

        ax_patch3.imshow(patches_at_gt_points[i])
        ax_patch3.set_title("Image at current\ntrue keypoint position")

        # Add legend to main plot
        true_patch = mpatches.Patch(color="green", label="True Position")
        inferred_patch = mpatches.Patch(color="blue", label="Inferred Position")
        ax_main.legend(
            handles=[true_patch, inferred_patch],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
        )

        # Adjust layout
        plt.tight_layout()

    # Create the animation
    anim = FuncAnimation(
        fig, animate, frames=len(video), interval=1000 / fps, blit=False
    )

    # Set up the FFmpeg writer
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)

    # Save as MP4
    anim.save(output_filename, writer=writer)
    plt.close(fig)
