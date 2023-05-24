import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

num_images = 9
prompt = ["a photograph of an astronaut riding a horse"] * num_images

# guidance_scale: adherence to prompt ~7-8.5 generally good
# num_inference_steps: ~quality, 50 is normal, less is faster
generator = torch.Generator("cuda").manual_seed(1024)
images = pipe(prompt, guidance_scale=7.5, num_inference_steps=5, generator=generator).images
grid = image_grid(images, rows=3, cols=3)

grid.save(f"res.png")
