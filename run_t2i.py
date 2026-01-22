import os
import gc
import sys
import time
import random
import string
import logging
from PIL import Image
from pathlib import Path

import torch
import requests
from diffusers.image_processor import PixArtImageProcessor

from painter.prompt import PromptGrid
from painter.summarizer import LocalSummarizer
from painter.helper import sdxl_automated_pipeline
from painter.painter import (
    SD15Painter,
    SDXLPainter,
    SDXLImg2ImgPainter,
    SDXLInpaintPainter,
)

# Setup
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

requests.post(
    "http://localhost:11434/api/generate", json={"model": "llama3", "keep_alive": 0}
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

model = "models/juggernautXL_ragnarokBy.safetensors"
width = 128 * 18
height = 128 * 4
cfgs = [3]
steps = [18]
grid_width = 1
grid_height = 1

path = "prompts/automatic"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
files_neg = [f for f in files if "NEG" in f]
files = [f for f in files if not "NEG" in f]
files_neg.sort()
files.sort()
names = [Path(f).stem for f in files]
poss = [open(os.path.join(path, f), "r", encoding="utf-8").read() for f in files]
negs = [open(os.path.join(path, f), "r", encoding="utf-8").read() for f in files_neg]



poss = ["medieval oil painting close-up of a wooden medieval writing desk filled with all kinds of stuff, close-up and birds-eye-view, full of many papers and books, large open calender, clock, time measurement, the whole image is primarily in yellow tones"]
negs = ["bad quality, worst quality, worst detail, sketch, text, signature"]
names = ["Doggenhof"]

sdxl_automated_pipeline(
    model,
    names,
    poss,
    negs,
    cfgs,
    steps,
    grid=(grid_height, grid_width),
    width=width,
    height=height,
    annotate=False,
    mode="1to1"
)
