from ....common.solver import Solver
import jax
import jax.numpy as jnp
import b3d
import genjax
from genjax import ChoiceMapBuilder as C
from .model import model_factory, get_likelihood, rr_log_trace

RENDERER_HYPERPARAMS = (
    b3d.chisight.dense.differentiable_renderer.DifferentiableRendererHyperparams(
        3, 1e-5, 1e-2, -1
    )
)


@genjax.gen
def gt_informed_triangle_proposal(gt_triangle, mindepth, maxdepth, fx, fy, cx, cy):
    """
    Given a triangle `gt_triangle = [A, B, C]`, where A, B, C are 3-vectors
    in the camera frame, propose a new triangle `new_triangle = [D, E, F]` in
    the camera frame that projects to approximately the same pixels as `gt_triangle`.
    """
    # jax.experimental.checkify.check(jnp.allclose(fx, fy), "fx != fy")
    A, B, C = gt_triangle

    # Origin of the camera frame
    O = jnp.zeros(3)

    # Step 1: sample a random depth along the ray from
    # the camera origin to each vertex of the triangle.
    depth1 = genjax.uniform(mindepth, maxdepth) @ "depth1"
    # depth2 = genjax.uniform(mindepth, maxdepth) @ "depth2"
    # depth3 = genjax.uniform(mindepth, maxdepth) @ "depth3"

    D = O + depth1 / jnp.linalg.norm(A - O) * (A - O)
    E = O + depth1 / jnp.linalg.norm(B - O) * (B - O)
    F = O + depth1 / jnp.linalg.norm(C - O) * (C - O)

    new_triangle = jnp.stack([D, E, F], axis=0)
    return new_triangle


def importance_sample_with_depth_in_partition(
    key, task_input, model, mindepth, maxdepth
):
    """
    triangle will be a 3x3 array [v1, v2, v3] (not normalized! at true pose!)

    P(depth | image, depth \in [mindepth, maxdepth) )
    w ~~ P( image, depth \in [mindepth, maxdepth) )
    """
    X_WC = task_input["camera_path"][0]
    triangle_W = task_input["cheating_info"]["triangle_vertices"]
    triangle_C = X_WC.inv().apply(triangle_W)
    r = task_input["renderer"]

    _, log_q_score, new_triangle_C = gt_informed_triangle_proposal.propose(
        key, (triangle_C, mindepth, maxdepth, r.fx, r.fy, r.cx, r.cy)
    )

    new_triangle_W = X_WC.apply(new_triangle_C)

    T = task_input["camera_path"].shape[0]
    trace, log_p_score = model.importance(
        key,
        C.d(
            {
                "triangle_vertices": new_triangle_W,
                "observed_rgbs": genjax.ChoiceMap.idx(
                    jnp.arange(T), C.n().at["observed_rgb"].set(task_input["video"])
                ),
            }
        ),
        (
            task_input["background_mesh"],
            task_input["triangle"]["color"],
            task_input["camera_path"],
        ),
    )

    return trace, log_p_score - log_q_score


class ImportanceSolver(Solver):
    def solve(self, task_input):
        partition = task_input["depth_partition"]
        key = jax.random.PRNGKey(1)
        renderer = task_input["renderer"]
        model = model_factory(renderer, get_likelihood(renderer), RENDERER_HYPERPARAMS)

        partition_starts = partition[:-1]
        partition_ends = partition[1:]

        @jax.jit
        def get_tr_and_score(k, s, e):
            tr, score = importance_sample_with_depth_in_partition(
                k, task_input, model, s, e
            )
            return (tr, score)

        trs_and_scores = [
            get_tr_and_score(k, s, e)
            for (k, s, e) in zip(
                jax.random.split(key, len(partition_starts)),
                partition_starts,
                partition_ends,
            )
        ]
        trs = [ts[0] for ts in trs_and_scores]
        joint_scores = jnp.array([ts[1] for ts in trs_and_scores])

        for i in range(len(trs)):
            if i in (50, 80):
                rr_log_trace(trs[i], task_input["renderer"], prefix=f"trace_{i}")

        return joint_scores - jax.scipy.special.logsumexp(joint_scores)
