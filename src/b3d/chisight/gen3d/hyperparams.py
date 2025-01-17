from genjax import Pytree


@Pytree.dataclass
class InferenceHyperparams(Pytree):
    n_poses: int = Pytree.static()

    pose_proposal_args: any = Pytree.static(
        # default_factory=(lambda: [(0.05, 1000.0), (0.02, 1500.0), (0.005, 2000.0)])
        default_factory=(lambda: [(0.05, 500.0), (0.02, 1000.0), (0.005, 1500.0)])
    )

    include_q_scores_at_top_level: bool = True
