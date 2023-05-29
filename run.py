import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import DPMSolverMultistepScheduler
import PIL
import requests
import os

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def dummy(images, **kwargs):
    return images, False

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

dif_prompts = [
"alien landscape, weird, strange, plants, ruins",
"lonely house, old, delapidated",
"selfie, woman, bedroom",
"group of people at festival, music, dance",
"large machine, inconceivable, unknown",
"aerial shot of woods, zoomed out",
"hotel room, luxurious",
"mountainous landscape, snow, clouds",
"robot, portrait, sci-fi",
"banker, suit",
"mass of wriggling worms"
]

dif_prompts = [
"small halfling man, merchant, middle-aged, older, some wrinkles, wide devious grin, slicked-back black hair, medieval, mischievous, colorful clothes, backpack with many things, goatee, stubble, inviting gesture, next to donkey, donkey in front of cart, full body"
]

dif_prompt_styles = [
"(masterpiece), High detail color photo, a professional photo, (realistic, photorealism:1. 5), (highest quality), (best shadow), ultra-high resolution, physics-based rendering, photo, realism, high contrast, 8k HD high definition detailed realistic, detailed, best quality, Nikon d850 film, cinestill 800",
"character sheet, concept design, contrast, style by kim jung gi, zabrocki, karlkka, jayison devadas, trending on artstation, 8k, ultra wide angle, pincushion lens effect",
"epic concept art by barlowe wayne, ruan jia and greg rutkowski, light effect, volumetric light, 3d, ultra clear detailed. octane render. 8k. dark green, light grey blue and golden colour scheme",
"award winning watercolor pen illustration, detailed, disney, isometric illustration, drawing, by Stephen Hillenburg, Matt Groening, Albert Uderzo",
"23rd century scientific schematics, blueprint, hyperdetailed vector technical documents, callouts, archviz, legend, patent registry",
"steampunk cybernetic biomechanical, 3d model, very coherent symmetrical artwork, unreal engine realistic render, 8 k, micro detail, intricate, elegant, highly detailed, centered, digital painting, artstation, smooth, sharp focus, illustration, artgerm, tomasz alen kopera, wlop",
"epic, by Andrew McCarthy, Navaneeth Unnikrishnan, Manuel Dietrich, photo realistic, 8 k, cinematic lighting, hd, atmospheric, hyperdetailed, trending on artstation, deviantart, photography, glow effect",
"photograph, photorealistic, vivid, sharp focus, reflection, refraction, sunrays, very detailed, intricate, intense cinematic composition"
]

dif_neg_prompts = [
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
    "bad anatomy, bad composition, ugly, abnormal, unrealistic, double, contorted, disfigured, malformed, amateur, extra, duplicate",
    "",
]

# guidance_scale: adherence to prompt ~7-8.5 generally good
# num_inference_steps: ~quality, 50 is normal, less is faster

#pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_ckpt("models/dreamshaper_6BakedVae.safetensors", torch_dtype=torch.float16)
#pipe = StableDiffusionPipeline.from_ckpt("models/revAnimated_v122.safetensors", torch_dtype=torch.float16)
#pipe = StableDiffusionPipeline.from_ckpt("models/chilloutmix_NiPrunedFp32Fix.safetensors", torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.safety_checker = dummy
pipe = pipe.to("cuda")

num_images = 12
prompt_n = 1
style_n = 1
neg_prompt_n = 1
for prompt_str in dif_prompts:
    for style_str in dif_prompt_styles:
        for neg_prompt_str in dif_neg_prompts:
            prompt = [prompt_str + ", " + style_str] * num_images
            neg_prompt = [neg_prompt_str] * num_images

            generator = torch.Generator("cuda").manual_seed(100)
            images = pipe(prompt, negative_prompt=neg_prompt, guidance_scale=7.5, num_inference_steps=20, generator=generator).images
            grid = image_grid(images, rows=3, cols=4)
            print("#"*150)
            print(prompt[0])

            grid.save(f"results/" + prompt_str.split(", ")[0].replace(" ", "_") + "_" + str(style_n) + neg_prompt_str.split(", ")[0].replace(" ", "_") + ".png")
            neg_prompt_n +=1
        style_n +=1
    prompt_n +=1

