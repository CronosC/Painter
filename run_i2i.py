import os
import gc
import random
import torch
from pathlib import Path
from PIL import Image

from painter.painter import SDXLImg2ImgPainter
from painter.prompt import PromptGrid

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "models/juggernautXL_ragnarokBy.safetensors"

path = "results"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for file in files:

    INPUT_IMAGE_PATH = path + "/" + file
    OUTPUT_PATH = "results/filtered"

    WIDTH = 128 * 5
    HEIGHT = 128 * 5
    STRENGTH = 0.35      
    CFG_SCALE = 7.0
    STEPS = 40
    SEED = 42
    GRID = (1, 1)    

    # Prompts
    POSITIVES = [
        "classical medieval (oil painting:2), (rough oil brushstrokes:2), expressive (painterly:3) strokes, impasto highlights, painterly, chiaroscuro lighting, waist-up portrait"
    ]

    NEGATIVES = [
        "photorealistic, nudity, nsfw, bad quality, text, signature, modern"
    ]
    name = Path(file).stem
    name = name.translate(str.maketrans({
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", '"': "'", "†": ""
    }))
    NAMES = [name]


    init_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    init_image = init_image.resize((WIDTH, HEIGHT))
    #WIDTH = init_image.width
    #HEIGHT = init_image.height

    prompts = PromptGrid(WIDTH, HEIGHT)
    prompts.add_variations(
        positives=POSITIVES,
        negatives=NEGATIVES,
        seeds=[SEED],
        cfgs=[CFG_SCALE],
        steps=[STEPS],
        names=NAMES
    )

    for p in prompts:
        p.image = init_image
        p.strength = STRENGTH


    painter = SDXLImg2ImgPainter(MODEL_PATH, output_dir=OUTPUT_PATH)
    results = painter.paint(prompts, grid=GRID, annotate=False, overwrite=False)


    del painter
    gc.collect()
    torch.cuda.empty_cache()
