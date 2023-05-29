import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import os

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
to_scale = Image.open("res.png").convert("RGB")
prompt = "tarsier monkey,(masterpiece), High detail color photo, a professional photo, (realistic, photorealism:1. 5), (highest quality), (best shadow), ultra-high resolution, physics-based rendering, photo, realism, high contrast, 8k HD high definition detailed realistic, detailed, best quality, Nikon d850 film, cinestill 800"

upscaled_image = pipeline(prompt=prompt, image=to_scale).images[0]
upscaled_image.save("upsampled_res.png")