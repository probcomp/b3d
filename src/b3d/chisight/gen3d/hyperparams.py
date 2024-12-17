from genjax import Pytree


@Pytree.dataclass
class InferenceHyperparams(Pytree):
    n_poses: int = Pytree.static()

    pose_proposal_args: any = Pytree.static(
        default_factory=(lambda: [(0.04, 1000.0), (0.02, 1500.0), (0.005, 2000.0)])
    )

    include_q_scores_at_top_level: bool = True