import os

scenario = "collide"
# for clip in ["first", "last", "avg", "gt"]:
for clip in ["gt"]:
    for all_scale in ["first_scale", "all_scale"]:
        print(f"--------------------{all_scale}--------------------")
        os.system(
            f"python test_b3d_tracking_batch.py --scenario {scenario} --clip {clip} --all_scale {all_scale}"
        )
