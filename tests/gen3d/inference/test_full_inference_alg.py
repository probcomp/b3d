import b3d
import b3d.chisight.gen3d.inference.inference as inference
import b3d.chisight.gen3d.model
import b3d.chisight.gen3d.settings
import b3d.chisight.gen3d.settings as settings
import b3d.io.data_loader
import jax
import jax.numpy as jnp
from b3d.chisight.gen3d.dataloading import (
    get_initial_state,
    load_object_given_scene,
    load_scene,
)
from b3d.chisight.gen3d.model import (
    get_new_state,
)
from tqdm import tqdm


def test_inference_alg_runs_and_looks_ok():
    scene_id = 49
    FRAME_RATE = 50
    OBJECT_INDEX = 1

    hyperparams = settings.hyperparams
    inference_hyperparams = settings.inference_hyperparams

    all_data, meshes, renderer, intrinsics, _ = load_scene(scene_id, FRAME_RATE)
    template_pose, model_vertices, model_colors = load_object_given_scene(
        all_data, meshes, renderer, OBJECT_INDEX
    )
    hyperparams["intrinsics"] = intrinsics
    hyperparams["vertices"] = model_vertices

    initial_state = get_initial_state(
        template_pose, model_vertices, model_colors, hyperparams
    )

    ### Test we can generate a trace ###
    key = jax.random.PRNGKey(0)
    og_trace, weight = inference.get_initial_trace(
        key, hyperparams, initial_state, all_data[0]["rgbd"], get_weight=True
    )
    assert (
        weight == og_trace.get_score()
    )  # Test that all addresses are constrained in this trace generation

    ### Test one inference step ###
    def gt_pose(T):
        return (
            all_data[T]["camera_pose"].inv() @ all_data[T]["object_poses"][OBJECT_INDEX]
        )

    trace, _ = inference.inference_step(
        jax.random.PRNGKey(26),
        og_trace,
        all_data[0]["rgbd"],
        inference_hyperparams,
        use_gt_pose=True,
        gt_pose=gt_pose(0),
    )

    assert (
        jnp.linalg.norm(get_new_state(trace)["pose"].position - gt_pose(0).position)
        < 0.004
    )

    ### Run inference, giving the ground truth pose as a option in the pose proposal grid ###
    trace = og_trace
    key = jax.random.PRNGKey(21)
    for T in tqdm(range(2)):
        key = b3d.split_key(key)
        trace, _ = inference.inference_step(
            jax.random.PRNGKey(26),
            trace,
            all_data[T]["rgbd"],
            inference_hyperparams,
            use_gt_pose=True,
            gt_pose=gt_pose(T),
        )

        assert (
            jnp.linalg.norm(get_new_state(trace)["pose"].position - gt_pose(T).position)
            < 0.007
        )

    ### Real inference run ###
    key = jax.random.PRNGKey(123)
    trace = og_trace
    for T in tqdm(range(2)):
        key = b3d.split_key(key)
        trace, _ = inference.inference_step(
            jax.random.PRNGKey(26),
            trace,
            all_data[T]["rgbd"],
            inference_hyperparams,
            use_gt_pose=False,
        )

        assert (
            jnp.linalg.norm(get_new_state(trace)["pose"].position - gt_pose(T).position)
            < 0.02
        )
