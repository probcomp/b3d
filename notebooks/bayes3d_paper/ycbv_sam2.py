# import os

# import b3d
# import numpy as np

# scene = 48

# FRAME_RATE = 50

# ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")


# if scene is None:
#     scenes = range(48, 60)
# elif isinstance(scene, int):
#     scenes = [scene]
# elif isinstance(scene, list):
#     scenes = scene

# scene_id = scenes[0]


# print(f"Scene {scene_id}")
# b3d.reload(b3d.io.data_loader)
# num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)

# # image_ids = [image] if image is not None else range(1, num_scenes, FRAME_RATE)
# image_ids = range(1, num_scenes + 1, FRAME_RATE)
# all_data = b3d.io.data_loader.get_ycbv_test_images(ycb_dir, scene_id, image_ids)


# import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# checkpoint = "../segment-anything-2/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# my_image = np.array(all_data[0]["rgbd"][..., :3])

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     predictor.set_image(my_image)
#     masks, _, _ = predictor.predict()
