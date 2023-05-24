import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "a highly detailed, stylized picture of a tasier, emphasized large eyes, watercolor, colorful"

image = pipe(prompt).images[0]
image.save(f"res.png")
