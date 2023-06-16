import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionUpscalePipeline
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPImageProcessor
from transformers import CLIPTokenizer
import PIL
import requests
import os
from compel import Compel
import random
import math
from io import BytesIO


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

class Painter:
    # guidance_scale: adherence to prompt ~7-8.5 generally good
    # num_inference_steps: ~quality, 50 is normal, less is faster
    def __init__(self, modelT2I):
        if modelT2I == 1:
            self.pipeT2I = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
            self.pipeI2I = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        elif modelT2I == 2:
            self.pipeT2I = StableDiffusionPipeline.from_ckpt("models/dreamshaper_6BakedVae.safetensors", torch_dtype=torch.float16)
            self.pipeI2I = StableDiffusionImg2ImgPipeline.from_ckpt("models/dreamshaper_6BakedVae.safetensors", torch_dtype=torch.float16)
            #self.pipeT2I.load_textual_inversion("embeddings/BadDream.pt")
            #self.pipeT2I.load_textual_inversion("embeddings/UnrealisticDream.pt")
            self.pipeT2I.load_textual_inversion("embeddings/FastNegativeEmbedding.pt")
        elif modelT2I == 3:
            self.pipeT2I = StableDiffusionPipeline.from_ckpt("models/revAnimated_v122.safetensors", torch_dtype=torch.float16)
            self.pipeI2I = StableDiffusionImg2ImgPipeline.from_ckpt("models/revAnimated_v122.safetensors", torch_dtype=torch.float16)
        elif modelT2I == 4:
            self.pipeT2I = StableDiffusionPipeline.from_ckpt("models/edgeOfRealism_eorV20Fp16BakedVAE.safetensors", torch_dtype=torch.float16)
            self.pipeI2I = StableDiffusionImg2ImgPipeline.from_ckpt("models/edgeOfRealism_eorV20Fp16BakedVAE.safetensors", torch_dtype=torch.float16)
            self.pipeT2I.load_textual_inversion("embeddings/FastNegativeEmbedding.pt")
        else:
            self.pipeT2I = StableDiffusionPipeline.from_ckpt("models/chilloutmix_NiPrunedFp16Fix.safetensors", torch_dtype=torch.float16)
            self.pipeI2I = StableDiffusionImg2ImgPipeline.from_ckpt("models/chilloutmix_NiPrunedFp16Fix.safetensors", torch_dtype=torch.float16)
            self.pipeT2I.load_textual_inversion("embeddings/bad-picture-chill-32v.pt")
            #self.pipeI2I.load_textual_inversion("embeddings/bad-picture-chill-32v.pt")

        self.pipeT2I.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeT2I.scheduler.config)
        # self.pipeT2I.feature_extractor = CLIPImageProcessor()# .from_config(self.pipeT2I.feature_extractor.config)
        # self.pipeT2I.tokenizer = CLIPTokenizer()#.from_config(self.pipeT2I.tokenizer.config)

        #self.pipeT2I.enable_model_cpu_offload()
        self.pipeT2I.enable_attention_slicing()
        self.pipeT2I.enable_vae_slicing()
        self.pipeT2I.enable_vae_tiling()
        self.pipeT2I.safety_checker = dummy
        self.pipeT2I.enable_xformers_memory_efficient_attention()
        self.pipeT2I = self.pipeT2I.to("cuda")


        self.pipeI2I.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeI2I.scheduler.config)

        self.pipeI2I.enable_attention_slicing()
        self.pipeI2I.safety_checker = dummy
        self.pipeI2I.enable_xformers_memory_efficient_attention()
        self.pipeI2I = self.pipeI2I.to("cuda")

        #self.pipeUpscale = StableDiffusionUpscalePipeline.from_pretrained(**self.pipeT2I.components, torch_dtype=torch.float16)
        '''self.pipeUpscale.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeUpscale.scheduler.config)
        self.pipeUpscale.enable_attention_slicing()
        self.pipeUpscale.safety_checker = dummy
        self.pipeUpscale.enable_xformers_memory_efficient_attention()
        self.pipeUpscale.enable_model_cpu_offload()'''
        #self.pipeUpscale = self.pipeUpscale.to("cuda")

        self.compel = Compel(tokenizer=self.pipeT2I.tokenizer, text_encoder=self.pipeT2I.text_encoder, truncate_long_prompts=True)

    def paint(self, prompts, neg_prompts, rows, cols, image=None, vary_seed_rows=True, vary_seed_cols=True, w=512, h=512, guide_scale=7.5, inf_steps=25, save_as=""):
        num_images = len(prompts)

        if vary_seed_rows and vary_seed_cols:
            generator = [torch.Generator(device="cuda").manual_seed(1001 + i) for i in range(0, num_images)]
        elif vary_seed_rows:
            generator = [torch.Generator(device="cuda").manual_seed(1001 + math.floor(i/cols)) for i in range(num_images)]
        elif vary_seed_cols:
            generator = [torch.Generator(device="cuda").manual_seed(1001 + (i%cols)) for i in range(num_images)]
        else:
            generator = [torch.Generator(device="cuda").manual_seed(1001) for i in range(num_images)]

        prompts_emb, neg_prompts_emb = self.prepare_embeds(prompts, neg_prompts)

        with torch.inference_mode():
            if image==None:
                images = self.pipeT2I(prompt_embeds=prompts_emb, negative_prompt_embeds=neg_prompts_emb, width=w, height=h, guidance_scale=guide_scale, num_inference_steps=inf_steps, generator=generator).images
            else:
                images = self.pipeI2I(prompt_embeds=prompts_emb, negative_prompt_embeds=neg_prompts_emb, image=image, strength=0.35, guidance_scale=guide_scale, generator=generator).images

        if save_as == "":
            save_as = prompts[0].split(",")[0].replace(" ", "_") + "_" + neg_prompts[0].split(",")[0].replace(" ", "_")
        grid = image_grid(images, rows=rows, cols=cols)
        grid.save(f"results/" + save_as + ".png")

    def paint_axis_from_file(self, base, neg_prompt, fileX, fileY, image=None, modifier_file=""):
        base_prompts = []
        neg_prompts = []
        x_axis = []
        y_axis = []

        with open("prompts/" + base, "r") as base_file:
            lines = base_file.readlines()
            random.shuffle(lines)
            for line in lines:
                base_prompts.append(line.replace("\n", ""))

        with open("prompts/" + neg_prompt, "r") as base_file:
            lines = base_file.readlines()
            for line in lines:
                neg_prompts.append(line.replace("\n", ""))

        if fileX != None:
            if isinstance(fileX, int):
                x_axis = [""]*fileX
                vary_seed_cols = True
            else:
                vary_seed_cols = False
                with open("prompts/" + fileX, "r") as X_file:
                    lines = X_file.readlines()
                    for line in lines:
                        x_axis.append(line.replace("\n", ""))
        else:
            x_axis.append("")

        if fileY != None:
            if isinstance(fileY, int):
                y_axis = [""]*fileY
                vary_seed_rows = True
            else:
                vary_seed_rows = False
                with open("prompts/" + fileY, "r") as Y_file:
                    lines = Y_file.readlines()
                    for line in lines:
                        y_axis.append(line.replace("\n", ""))
        else:
            y_axis.append("")

        if modifier_file != "":
            with open("prompts/" + modifier_file, "r") as M_file:
                lines = M_file.readlines()
                for line in lines:
                    mod_str = line.replace("\n", "")


        cols = len(x_axis)
        rows = len(y_axis)
        n_max = len(base_prompts)
        n = 1
        for prompt in base_prompts:
            print(str(n) + "/" + str(n_max) + ": " + prompt)
            fin_prompts, fin_neg_prompts = self.prompt_pipeline_axis(prompt, neg_prompts, x_axis, y_axis, modifier=mod_str)
            self.paint(fin_prompts, fin_neg_prompts, rows, cols, image=image, vary_seed_cols=vary_seed_cols, vary_seed_rows=vary_seed_rows)
            n = n+1

    def prompt_pipeline_axis(self, base_prompt, neg_prompts, x_axis, y_axis, modifier=""):
        prompts = self.construct_prompt_strings_axis(base_prompt, x_axis, y_axis, modifier=modifier)

        i = 0
        while len(neg_prompts) < len(prompts):
            neg_prompts.append(neg_prompts[i])
            i+=1
            i=i%len(neg_prompts)

        return prompts, neg_prompts

    def construct_prompt_strings_axis(self, base_prompt, x_axis, y_axis, modifier=""):
        prompts = []
        for y in y_axis:
            for x in x_axis:
                prompts.append(self.construct_prompt_string_axis(base_prompt, x, y, modifier=modifier))
        return prompts

    def construct_prompt_string_axis(self, base_prompt, x_axis, y_axis, modifier=""):
        if modifier != "":
            modifier = ", " + modifier
        return base_prompt + ", " + x_axis + ", " + y_axis + modifier

    def prepare_embeds(self, prompts, negative_prompts):
        prompts_emb = self.compel(prompts)
        neg_prompts_emb = self.compel(negative_prompts)
        return prompts_emb, neg_prompts_emb

'''
image_path = "https://image.t"
response = requests.get(image_path)
init_image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((344, 512))
'''

painter = Painter(4)
painter.paint_axis_from_file("Prompts", "Negative", 3, 3, modifier_file="HQ")

