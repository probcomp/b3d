import torch
import b3d
import os
import numpy as np
import time
import argparse



parser = argparse.ArgumentParser("r3d_to_video_input")
parser.add_argument("input", help=".r3d File", type=str)
args = parser.parse_args()



path = args.input

# # Load date
# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/royce_static_to_dynamic.r3d.video_input.npz",
# )
video_input = b3d.io.VideoInput.load(path)
frames = np.array(video_input.rgb)[::4]
print(frames.shape)

device = 'cuda'
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)

video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

grid_size = 70
t0 = time.time()
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
t1 = time.time()

print("Cotracker took ", t1-t0, " seconds")

# t0 = time.time()
# pred_tracks, pred_visibility = cotracker(video, grid_siize=grid_size) # B T N 2,  B T N 1
# t1 = time.time()

# print("Cotracker took ", t1-t0, " seconds")

# from cotracker.utils.visualizer import Visualizer

# vis = Visualizer(save_dir=".", pad_value=120, linewidth=3)
# vis.visualize(video, pred_tracks, pred_visibility)

pred_tracks_ = pred_tracks.cpu().numpy()
pred_visibility_ = pred_visibility.cpu().numpy()
np.savez(path + "cotracker_output.npz", pred_tracks=pred_tracks_, pred_visibility=pred_visibility_)


