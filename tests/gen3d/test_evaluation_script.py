from src.b3d.chisight.gen3d.run_ycbv_evaluation import run_tracking


def test_run_tracking():
    run_tracking(scene=49, object=1, max_n_frames=5)
