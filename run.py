import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import DPMSolverMultistepScheduler
import PIL
import requests
import os

torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def download_image(url):
   image = PIL.Image.open(requests.get(url, stream=True).raw)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB")
   return image

#pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_ckpt("models/dreamshaper_6BakedVae.safetensors", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

num_images = 5
prompt = ["emoji, smiley, (tarsier monkey), chat, icon, symbol (masterpiece), High detail color photo, a professional photo, (realistic, photorealism:1. 5), (highest quality), (best shadow), ultra-high resolution, physics-based rendering, photo, realism, high contrast, 8k HD high definition detailed realistic, detailed, best quality, Nikon d850 film, cinestill 800"] * num_images
neg_prompt = ["(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"] * num_images
# guidance_scale: adherence to prompt ~7-8.5 generally good
# num_inference_steps: ~quality, 50 is normal, less is faster
generator = torch.Generator("cuda").manual_seed(1234)
images = pipe(prompt, negative_prompt=neg_prompt, guidance_scale=7.5, num_inference_steps=30, generator=generator).images
grid = image_grid(images, rows=1, cols=5)

grid.save(f"res.png")
