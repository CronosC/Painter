import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

test_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
device = "cuda"
model_id_or_path = "stabilityai/stable-diffusion-2-1"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

def image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((768, 512))
    return image

def generate_image(image, prompt, strength=0.95, scale=7.5, steps=50):
    return pipe(prompt=prompt, image=image, num_inference_steps=steps, strength=strength, guidance_scale=scale).images

def save_image(image, name):
    image[0].save(name + ".png")


prompt = "a fantastic mountainscape, grand, breathtaking, digital art"
init_img = image_from_url(test_url)

img = generate_image(init_img, prompt)
save_image(img, "img2imgTesting")