#!/usr/bin/env python
import fire

def make_visual(scene=None, object=None, debug=False):
    import b3d
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import jax
    from tqdm import tqdm
    from b3d import Pose, Mesh
    import rerun as rr
    import genjax
    import os
    import genjax
    from b3d.modeling_utils import uniform_discrete, uniform_pose, gaussian_vmf
    from collections import namedtuple
    from genjax import Pytree
    import b3d
    from b3d.bayes3d.enumerative_proposals import gvmf_and_select_best_move
    from tqdm import tqdm
    from IPython import embed
    import fire

    import importlib
    importlib.reload(b3d.mesh)
    importlib.reload(b3d.io.data_loader)
    importlib.reload(b3d.utils)
    importlib.reload(b3d.renderer.renderer_original)

    FRAME_RATE = 50

    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")

    b3d.rr_init()


    scene_id = scene

    num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)
    image_ids = range(1, num_scenes, FRAME_RATE)
    all_data = b3d.io.get_ycbv_test_images(ycb_dir, scene_id, image_ids)

    height, width = all_data[0]["rgbd"].shape[:2]
    fx,fy,cx,cy = all_data[0]["camera_intrinsics"]
    scaling_factor = 1.0
    renderer = b3d.renderer.renderer_original.RendererOriginal(
        width * scaling_factor, height * scaling_factor, fx * scaling_factor, fy * scaling_factor, cx * scaling_factor, cy * scaling_factor, 0.01, 2.0
    )

    initial_object_poses = all_data[0]["object_poses"]
    object_indices = [object] if object is not None else range(len(initial_object_poses))
    for object in object_indices:

        print(f"Generative video for Scene {scene_id} and Object {object}")
        inferred_poses_data = jnp.load(f"SCENE_{scene_id}_OBJECT_INDEX_{object}_POSES.npy.npz", allow_pickle=True)
        inferred_poses = Pose(inferred_poses_data["position"], inferred_poses_data["quaternion"])

        id = all_data[0]["object_types"][object]
        mesh = Mesh.from_obj_file(
            os.path.join(
                ycb_dir,
                f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply'
            )
        ).scale(0.001)

        video_frames = []
        for t in tqdm(range(len(all_data))):
            actual_rgbd = all_data[t]["rgbd"]
            b3d_inferred_rgbd = renderer.render_rgbd_from_mesh(mesh.transform(inferred_poses[t]))


            import glob
            image_id = image_ids[t]
            foundation_pose_results_dir = "FoundationPose_every_50_frames_gt_init"
            filename = os.path.join(
                foundation_pose_results_dir,
                str(scene_id).rjust(6, "0"),
                f"{all_data[0]["object_types"][object]+1}",
                "ob_in_cam",
                str(image_id).rjust(6, "0") + ".txt"
            )
            import numpy as np
            foundation_predicted_pose = Pose.from_matrix(jnp.array(np.loadtxt(filename)))
            foundation_inferred_rgbd = renderer.render_rgbd_from_mesh(mesh.transform(foundation_predicted_pose))

            actual_viz = b3d.viz_rgb(actual_rgbd)
            b3d_inferred_viz = b3d.viz_rgb(b3d_inferred_rgbd)
            foundation_inferred_viz = b3d.viz_rgb(foundation_inferred_rgbd)

            alpha = 0.75
            video_frames.append(
                b3d.multi_panel(
                    [actual_viz, b3d.overlay_image(actual_viz,b3d_inferred_viz, alpha=alpha), b3d.overlay_image(actual_viz,foundation_inferred_viz,alpha=alpha)],
                    labels=["Input", "Gen3D", "FoundationPose"],
                    label_fontsize=55
                )
            )
        b3d.make_video_from_pil_images(
            video_frames,
            f"video_scene_{scene_id}_object_{object}.mp4"
        )




if __name__ == "__main__":
    fire.Fire(make_visual)
