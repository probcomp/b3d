import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import trimesh

import b3d
from b3d import Pose


def render_rgbd_many(renderer, vertices, faces, attributes):
    return renderer.render_many(
        vertices, faces, jnp.concatenate([attributes, vertices[..., -1:]], axis=-1)
    )


def make_mesh_object_library(mesh_path):
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)
    return object_library


def test_distance_invariance(renderer, models, model_names, model_args_dicts):
    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    object_library = make_mesh_object_library(mesh_path)

    cam_y_distance = 0.8

    def linear_pose_from_points(points1, points2, alpha):
        return Pose.from_position_and_target(
            jnp.array([0.0, -cam_y_distance, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation((1 - alpha) * points1 + alpha * points2)

    distances = jnp.linspace(0, 1.5, 4) - 0.3
    num_points = 4001
    distance = 0.25
    horizontal_range = 0.1
    observed_images = []

    for d_ind, distance in enumerate(distances):
        gt_pose = b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation(jnp.array([0, distance, 0]))
        transformed_vertices = gt_pose.apply(object_library.vertices)[None, ...]
        images = render_rgbd_many(
            renderer,
            transformed_vertices,
            object_library.faces,
            jnp.tile(object_library.attributes, (1, 1, 1)),
        )
        gt_image = images[0, ..., :3]
        gt_image_depth = np.array(images[0, ..., 3])
        gt_image_depth[gt_image_depth == 0] = 10.0
        gt_image_depth = jnp.array(gt_image_depth)
        observed_images.append(
            jnp.concatenate([gt_image, gt_image_depth[..., None]], axis=-1)
        )

    vec_fun = jax.vmap(linear_pose_from_points, (None, None, 0))

    # sweep y from -0.25 to 1.5
    # plt.figure()
    model_scores = []

    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 7))
    for model_ind, model in enumerate(models):
        ax = axes[model_ind]
        model_scores_distances = []
        for d_ind, distance in enumerate(distances):
            point1 = jnp.array([-horizontal_range, distance, 0])
            point2 = jnp.array([horizontal_range, distance, 0])
            alphas = jnp.linspace(0, 1, num_points)
            linear_poses_batches = vec_fun(point1, point2, alphas).split(10)
            logpdfs = []
            translations = []
            for linear_poses in linear_poses_batches:
                transformed_vertices = jax.vmap(
                    lambda i: linear_poses[i].apply(object_library.vertices)
                )(jnp.arange(len(linear_poses)))
                rendered_imgs = render_rgbd_many(
                    renderer,
                    transformed_vertices,
                    object_library.faces,
                    jnp.tile(object_library.attributes, (len(linear_poses), 1, 1)),
                )
                logpdfs.append(
                    model(
                        observed_images[d_ind],
                        rendered_imgs,
                        model_args_dicts[model_ind],
                    )[0]
                )
                translations.append(linear_poses.pos[:, 0])
            logpdfs = jnp.concatenate(logpdfs)
            model_scores_distances.append(logpdfs)
            translations = jnp.concatenate(translations)
            ax.plot(
                translations,
                b3d.normalize_log_scores(logpdfs),
                alpha=0.7,
                linewidth=4,
                label=str(distance + cam_y_distance) + " (m)",
            )
        model_scores.append(model_scores_distances)
        title = "Posterior conditioned on two two-object scene\nModel: " + str(
            model_names[model_ind]
        )
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(fontsize=20)
        ax.set_xlabel("Horizontal distance from GT (meters)", fontsize=20)
        ax.grid(True)
        if model_ind == 0:
            ax.set_ylabel("Posterior Probability", fontsize=20)
        else:
            ax.yaxis.set_ticklabels([])
        ax.set_ylim(0.0, 0.008)
    fig.tight_layout()
    fig.savefig("posterior_invariance_plots/test_distance_invariance.png")
    model_scores = jnp.array(model_scores)

    ## Draw samples to show posterior

    key = jax.random.PRNGKey(0)
    apply_vec = jax.vmap(lambda pose, vertices: pose.apply(vertices), (0, None))
    num_samples = 10
    for dist_ind, distance in enumerate(distances):
        fig, axes = plt.subplots(
            len(models), num_samples, figsize=(num_samples + 2, len(models) + 1)
        )
        fig.suptitle(
            "Posterior samples - distance: " + str(distance + cam_y_distance) + " (m)",
            fontsize=12,
        )

        point1 = jnp.array([-horizontal_range, distance, 0])
        point2 = jnp.array([horizontal_range, distance, 0])
        alphas = jnp.linspace(0, 1, num_points)
        vec_fun = jax.vmap(linear_pose_from_points, (None, None, 0))
        linear_poses = vec_fun(point1, point2, alphas)

        rgbd_gt = render_rgbd_many(
            renderer,
            jnp.array(
                [linear_poses[len(linear_poses) // 2].apply(object_library.vertices)]
            ),
            object_library.faces,
            jnp.tile(object_library.attributes, (1, 1, 1)),
        )
        rgb_gt = rgbd_gt[0, ..., :3]

        for model_ind, _ in enumerate(models):
            scores = model_scores[model_ind, dist_ind]

            samples = jax.random.categorical(key, scores, shape=(num_samples,))

            rgbd_im = render_rgbd_many(
                renderer,
                apply_vec(linear_poses[samples], jnp.array(object_library.vertices)),
                object_library.faces,
                jnp.tile(object_library.attributes, (num_samples, 1, 1)),
            )

            for i in range(len(samples)):
                axes[model_ind, i].imshow((rgbd_im[i, ..., :3] + rgb_gt) / 2)
                # axes[model_ind, i].axis('off')
                axes[model_ind, i].tick_params(
                    left=False, labelleft=False, bottom=False, labelbottom=False
                )

            for ax, row in zip(axes[:, 0], model_names):
                ax.set_ylabel(row, rotation=90)
                ax.tick_params(
                    left=False, labelleft=False, bottom=False, labelbottom=False
                )
        # fig.tight_layout()
        fig.savefig(
            f"posterior_invariance_plots/test_distance_invariance_samples_{distance + cam_y_distance}m.png"
        )


def test_resolution_invariance(renderers, models, model_names, model_args_dicts):
    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    object_library = make_mesh_object_library(mesh_path)

    def linear_pose_from_points(points1, points2, alpha):
        return b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation((1 - alpha) * points1 + alpha * points2)

    num_points = 4001
    distance = 0.25
    observed_images = []

    for r_ind, renderer in enumerate(renderers):
        gt_pose = b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation(jnp.array([0, distance, 0]))
        transformed_vertices = gt_pose.apply(object_library.vertices)[None, ...]
        images = render_rgbd_many(
            renderer,
            transformed_vertices,
            object_library.faces,
            jnp.tile(object_library.attributes, (1, 1, 1)),
        )
        gt_image = images[0, ..., :3]
        gt_image_depth = np.array(images[0, ..., 3])
        gt_image_depth[gt_image_depth == 0] = 10.0
        gt_image_depth = jnp.array(gt_image_depth)
        observed_images.append(
            jnp.concatenate([gt_image, gt_image_depth[..., None]], axis=-1)
        )

    vec_fun = jax.vmap(linear_pose_from_points, (None, None, 0))

    # sweep y from -0.25 to 1.5
    plt.figure()
    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 7))
    for model_ind, model in enumerate(models):
        ax = axes[model_ind]
        for r_ind, renderer in enumerate(renderers):
            horizontal_range = 0.1
            point1 = jnp.array([-horizontal_range, distance, 0])
            point2 = jnp.array([horizontal_range, distance, 0])
            alphas = jnp.linspace(0, 1, num_points)
            linear_poses_batches = vec_fun(point1, point2, alphas).split(10)
            logpdfs = []
            translations = []
            for linear_poses in linear_poses_batches:
                transformed_vertices = jax.vmap(
                    lambda i: linear_poses[i].apply(object_library.vertices)
                )(jnp.arange(len(linear_poses)))
                rendered_imgs = render_rgbd_many(
                    renderer,
                    transformed_vertices,
                    object_library.faces,
                    jnp.tile(object_library.attributes, (len(linear_poses), 1, 1)),
                )
                logpdfs.append(
                    model(
                        observed_images[r_ind],
                        rendered_imgs,
                        model_args_dicts[model_ind],
                    )[0]
                )
                translations.append(linear_poses.pos[:, 0])
            logpdfs = jnp.concatenate(logpdfs)
            translations = jnp.concatenate(translations)
            ax.plot(
                translations,
                b3d.normalize_log_scores(logpdfs),
                alpha=0.7,
                linewidth=4,
                label=str(renderer.height) + "x" + str(renderer.width) + " pixels",
            )
        title = "Posterior conditioned on two two-object scene\nModel: " + str(
            model_names[model_ind]
        )
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(fontsize=20)
        ax.set_xlabel("Horizontal distance from GT (meters)", fontsize=20)
        ax.grid(True)
        if model_ind == 0:
            ax.set_ylabel("Posterior Probability", fontsize=20)
        else:
            ax.yaxis.set_ticklabels([])
        ax.set_ylim(0.0, 0.008)
    fig.tight_layout()
    fig.savefig("posterior_invariance_plots/test_resolution_invariance.png")

    plt.figure()
    fig, axes = plt.subplots(1, len(renderers), figsize=(10 * len(renderers), 7))
    for r_ind, renderer in enumerate(renderers):
        ax = axes[r_ind]
        ax.set_title(
            "conditioned image: \nresolution: "
            + str(renderer.width)
            + " x "
            + str(renderer.height)
            + "pixels",
            fontsize=20,
        )
        ax.imshow(observed_images[r_ind][..., :3])
        ax.axis("off")
    fig.savefig("posterior_invariance_plots/test_resolution_invariance_renderings.png")


def test_noise_invariance(renderer, models, model_names, model_args_dicts):
    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    object_library = make_mesh_object_library(mesh_path)

    def linear_pose_from_points(points1, points2, alpha):
        return b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation((1 - alpha) * points1 + alpha * points2)

    num_points = 4001
    distance = 0.25
    observed_images = []
    key = jax.random.PRNGKey(0)

    num_noise_points = 3
    rgb_noise_levels = np.linspace(0, 0.2, num_noise_points)
    depth_noise_levels = np.linspace(0, 0.2, num_noise_points)

    for n_ind, (rgb_noise_level, depth_noise_level) in enumerate(
        zip(rgb_noise_levels, depth_noise_levels)
    ):
        gt_pose = b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation(jnp.array([0, distance, 0]))
        transformed_vertices = gt_pose.apply(object_library.vertices)[None, ...]
        images = render_rgbd_many(
            renderer,
            transformed_vertices,
            object_library.faces,
            jnp.tile(object_library.attributes, (1, 1, 1)),
        )
        gt_image = images[0, ..., :3]
        gt_image = (
            jax.random.normal(key, shape=gt_image.shape) * rgb_noise_level + gt_image
        )
        gt_image_depth = np.array(images[0, ..., 3])
        gt_image_depth[gt_image_depth == 0] = 10.0
        gt_image_depth = jnp.array(gt_image_depth)
        gt_image_depth = (
            jax.random.normal(key, shape=gt_image_depth.shape) * depth_noise_level
            + gt_image_depth
        )
        observed_images.append(
            jnp.concatenate([gt_image, gt_image_depth[..., None]], axis=-1)
        )

    vec_fun = jax.vmap(linear_pose_from_points, (None, None, 0))
    # sweep y from -0.25 to 1.5
    plt.figure()
    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 7))
    for model_ind, model in enumerate(models):
        ax = axes[model_ind]
        for n_ind, (rgb_noise_level, depth_noise_level) in enumerate(
            zip(rgb_noise_levels, depth_noise_levels)
        ):
            horizontal_range = 0.1
            point1 = jnp.array([-horizontal_range, distance, 0])
            point2 = jnp.array([horizontal_range, distance, 0])
            alphas = jnp.linspace(0, 1, num_points)
            linear_poses_batches = vec_fun(point1, point2, alphas).split(10)
            logpdfs = []
            translations = []
            for linear_poses in linear_poses_batches:
                transformed_vertices = jax.vmap(
                    lambda i: linear_poses[i].apply(object_library.vertices)
                )(jnp.arange(len(linear_poses)))
                rendered_imgs = render_rgbd_many(
                    renderer,
                    transformed_vertices,
                    object_library.faces,
                    jnp.tile(object_library.attributes, (len(linear_poses), 1, 1)),
                )
                logpdfs.append(
                    model(
                        observed_images[n_ind],
                        rendered_imgs,
                        model_args_dicts[model_ind],
                    )[0]
                )
                translations.append(linear_poses.pos[:, 0])
            logpdfs = jnp.concatenate(logpdfs)
            translations = jnp.concatenate(translations)
            im_dim_str = (
                "$\sigma_c$: "
                + str(rgb_noise_level)
                + ", $\sigma_d$: "
                + str(depth_noise_level)
            )
            ax.plot(
                translations,
                b3d.normalize_log_scores(logpdfs),
                alpha=0.7,
                linewidth=4,
                label=im_dim_str,
            )
        title = "Posterior conditioned on two two-object scene\nModel: " + str(
            model_names[model_ind]
        )
        ax.set_title(title, fontsize=25)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(fontsize=20)
        ax.set_xlabel("Horizontal distance from GT (meters)", fontsize=20)
        ax.grid(True)
        if model_ind == 0:
            ax.set_ylabel("Posterior Probability", fontsize=20)
        else:
            ax.yaxis.set_ticklabels([])
        ax.set_ylim(0.0, 0.008)
    fig.tight_layout()
    fig.savefig("posterior_invariance_plots/test_noise_invariance.png")

    plt.figure()
    fig, axes = plt.subplots(
        1, len(rgb_noise_levels), figsize=(10 * len(rgb_noise_levels), 7)
    )
    for noise_ind, (rgb_noise_level, depth_noise_level) in enumerate(
        zip(rgb_noise_levels, depth_noise_levels)
    ):
        ax = axes[noise_ind]
        ax.set_title(
            "conditioned image: \ncolor $\sigma$: "
            + str(rgb_noise_level)
            + ", depth $\sigma$: "
            + str(depth_noise_level),
            fontsize=20,
        )
        ax.imshow(jnp.clip(observed_images[noise_ind], 0, 1))
        ax.axis("off")
    fig.savefig("posterior_invariance_plots/test_noise_invariance_renderings.png")


def test_mode_volume(renderer, models, model_names, model_args_dicts):
    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    object_library = make_mesh_object_library(mesh_path)

    def linear_pose_from_points(points1, points2, alpha):
        return b3d.Pose.from_position_and_target(
            jnp.array([0.0, -0.8, 0.1]), jnp.zeros(3)
        ).inv() @ b3d.Pose.from_translation((1 - alpha) * points1 + alpha * points2)

    point1 = jnp.array([-0.05, -0.25, 0])
    point2 = jnp.array([0.2, 1.5, 0])

    vec_fun = jax.vmap(linear_pose_from_points, (None, None, 0))
    alphas = jnp.linspace(-0.05, 1.5, 6)
    linear_poses = vec_fun(point1, point2, alphas)
    transformed_vertices = jax.vmap(
        lambda i: linear_poses[i].apply(object_library.vertices)
    )(jnp.arange(len(linear_poses)))

    rendered_imgs = render_rgbd_many(
        renderer,
        transformed_vertices,
        object_library.faces,
        jnp.tile(object_library.attributes, (len(linear_poses), 1, 1)),
    )

    gt_near_ind = len(alphas) // 6
    gt_far_ind = 4 * len(alphas) // 6
    gt_image = rendered_imgs[gt_near_ind, ..., :3] + rendered_imgs[gt_far_ind, ..., :3]
    gt_image_depth = np.array(
        rendered_imgs[gt_near_ind, ..., 3] + rendered_imgs[gt_far_ind, ..., 3]
    )
    gt_image_depth[gt_image_depth == 0] = 10.0
    gt_image_depth = jnp.array(gt_image_depth)
    observed_image = jnp.concatenate([gt_image, gt_image_depth[..., None]], axis=-1)

    # save plotted image renders
    plot_ims = [observed_image[..., :3]]
    selected_ims = rendered_imgs[jnp.array([0, gt_near_ind, gt_far_ind, -1]), ...]
    plot_ims += [*selected_ims[..., :3]]
    plot_ims = jnp.stack(plot_ims)
    labels = ["Conditioned Im", "Near", "Pose A", "Pose B", "Far"]

    f, axarr = plt.subplots(1, len(plot_ims), figsize=(31, 7))
    for ind, im in enumerate(plot_ims):
        axarr[ind].imshow(im)
        axarr[ind].set_xticklabels([])
        axarr[ind].set_xticks([])
        axarr[ind].set_yticklabels([])
        axarr[ind].set_yticks([])
        axarr[ind].set_title(labels[ind], fontsize=30, pad=20)
    f.savefig("posterior_invariance_plots/mode_volume_test_renders.png")

    # generate posterior plot
    alphas = jnp.linspace(-0.05, 1.5, 2001)
    linear_poses = vec_fun(point1, point2, alphas)
    fig, axes = plt.subplots(1, 1, figsize=(31, 9))

    for model_ind, model in enumerate(models):
        linear_poses_batches = vec_fun(point1, point2, alphas).split(10)
        logpdfs = []
        translations = []
        for linear_poses_batch in linear_poses_batches:
            transformed_vertices = jax.vmap(
                lambda i: linear_poses_batch[i].apply(object_library.vertices)
            )(jnp.arange(len(linear_poses_batch)))
            rendered_imgs = render_rgbd_many(
                renderer,
                transformed_vertices,
                object_library.faces,
                jnp.tile(object_library.attributes, (len(linear_poses_batch), 1, 1)),
            )
            logpdfs.append(
                model(observed_image, rendered_imgs, model_args_dicts[model_ind])[0]
            )
            translations.append(linear_poses_batch.pos[:, 0])
        logpdfs = jnp.concatenate(logpdfs)
        translations = jnp.concatenate(translations)
        axes.plot(
            linear_poses.pos[:, 2],
            b3d.normalize_log_scores(logpdfs),
            alpha=0.7,
            linewidth=4,
            label=model_names[model_ind],
        )

    num_ticks = 3
    xticks = list(np.linspace(0.35, 3.25, num_ticks))
    gt_near_ind = len(alphas) // 6
    gt_far_ind = 4 * len(alphas) // 6
    axes.set_xticks(
        xticks
        + [
            linear_poses.pos[gt_near_ind, 2],
            linear_poses.pos[gt_far_ind, 2],
            linear_poses.pos[0, 2],
            linear_poses.pos[-1, 2],
        ]
    )
    axes.set_xticklabels(
        ["%.1f" % f for f in xticks] + ["Pose A", "Pose B", "Near", "Far"], fontsize=30
    )
    axes.tick_params(axis="y", labelsize=25)
    title = "Scene object posterior conditioned on two mugs"
    axes.set_title(title, fontsize=40)
    axes.legend(fontsize=30)
    axes.set_xlabel("Distance from camera (meters)", fontsize=30)
    axes.set_ylabel("Posterior Probability", fontsize=30)
    axes.grid()
    fig.savefig("posterior_invariance_plots/mode_volume_test.png")


# this plot is alredy implemented in a separate existing test
def mug_posterior_peakiness_samples(renderer, models, model_names, model_args_dicts):
    # TODO: fill in publication-ready code for mug handle posterior angles
    # Case 1: mug handle is showing (posterior on angle should be delta function)
    # Case 2: mug handle is hidden (posterior should be fully cover angles that occlude handle from viewer)
    return None
