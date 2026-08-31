import io
import os

import jax
import jax.numpy as jnp
import numpy
import numpy as np
import rerun as rr
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    DetrFeatureExtractor,
    DetrForSegmentation,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)
from transformers.models.detr.feature_extraction_detr import rgb_to_id

import b3d

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

video_input = b3d.io.VideoInput.load(
    os.path.join(
        b3d.get_root_path(),
        # "assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz"
        "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz",
    )
)


image = Image.fromarray(np.array(video_input.rgb[0]))
image.save("test.png")


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
    target_sizes=[image.size[::-1]],
)


PORT = 8812
rr.init("real")
rr.connect(addr=f"127.0.0.1:{PORT}")

rr.log("image", rr.Image(np.array(image)))


rr.init("segmentation")
rr.connect("127.0.0.1:8812")

# Load date
# path = os.path.join(b3d.get_root_path(),
# "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
# video_input = b3d.io.VideoInput.load(path)

video_input = b3d.io.VideoInput.load(
    os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz",
        # "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
    )
)


feature_extractor = DetrFeatureExtractor.from_pretrained(
    "facebook/detr-resnet-50-panoptic"
)
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")


image = Image.fromarray(np.array(video_input.rgb[120 * 3]))

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
panoptic_seg_id_resize = jax.image.resize(
    panoptic_seg_id, (image_jnp.shape[0], image_jnp.shape[1]), "nearest"
)

rr.log("img", rr.Image(image_jnp))
rr.log("img/depth", rr.DepthImage(panoptic_seg_id_resize))
