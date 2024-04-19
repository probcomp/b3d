import io
import requests
from PIL import Image
import torch
import numpy
import jax.numpy as jnp
import numpy as np
import jax
import rerun as rr

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id


rr.init("segmentation")
rr.connect("127.0.0.1:8812")

import os 
import b3d

# Load date
# path = os.path.join(b3d.get_root_path(),
# "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
# video_input = b3d.VideoInput.load(path)

video_input = b3d.VideoInput.load(os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz"
# "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
))


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")


image = Image.fromarray(np.array(video_input.rgb[120*3]))

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

# the segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
# retrieve the ids corresponding to each mask
panoptic_seg_id = jnp.array(rgb_to_id(panoptic_seg))

image_jnp = jnp.array(image)
panoptic_seg_id_resize = jax.image.resize(panoptic_seg_id, (image_jnp.shape[0], image_jnp.shape[1]),
 "nearest")

rr.log("img", rr.Image(image_jnp))
rr.log("img/depth", rr.DepthImage(panoptic_seg_id_resize))