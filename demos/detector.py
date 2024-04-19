

import requests
from PIL import Image
import torch
import b3d
import os

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

video_input = b3d.VideoInput.load(os.path.join(b3d.get_root_path(),
# "assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz"
"assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
))

import numpy as np
image =Image.fromarray(np.array(video_input.rgb[0]))
image.save("test.png")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

text = "objects"
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)


import rerun as rr
PORT = 8812
rr.init("real")
rr.connect(addr=f'127.0.0.1:{PORT}')

rr.log("image", rr.Image(np.array(image)))