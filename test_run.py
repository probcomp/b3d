import os

scenario = "collide"
# for clip in ["first", "last", "avg", "gt"]:
#     os.system(f"python test_b3d_tracking_batch_multi_iter.py --scenario {scenario} --clip {clip}")

for clip in ["gt"]:
    for all_scale in ["first_scale", "all_scale"]:
        os.system(
            f"python test_b3d_tracking_batch_multi_iter.py --scenario {scenario} --clip {clip} --all_scale {all_scale}"
        )
