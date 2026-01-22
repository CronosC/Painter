import random
import itertools
import torch
import gc
from painter.prompt import PromptGrid
from painter.painter import (
    SD15Painter,
    SDXLPainter,
    SDXLImg2ImgPainter,
    SDXLInpaintPainter,
)

def apply_random_weights(data, chance=0.8, w_min=0.5, w_max=1.5):
    out = []
    for outer in data:
        new_outer = []
        for s in outer:
            parts = s.split(", ")
            if parts and random.random() < chance:
                i = random.randrange(len(parts))
                w = round(random.uniform(w_min, w_max), 2)
                parts[i] = f"({parts[i]}:{w})"
            new_outer.append(", ".join(parts))
        out.append(new_outer)
    return out

def sdxl_automated_pipeline(model, names, poss, negs, cfgs, steps, grid=(1,1), seeds=None, width=1024, height=1024, annotate=True, mode="1to1"):        
    prompts = PromptGrid(width, height, pos_dropout=0.0, neg_dropout=0.0)
    if mode == "crossproduct":
        prompts.add_variations(
            positives=poss, 
            negatives=negs, 
            seeds=seeds, 
            cfgs=cfgs, 
            steps=steps,
            names = names,
        )
    elif mode == "1to1":
        lists = [poss, negs, names, seeds or [None], cfgs, steps]
        max_len = max(len(lst) for lst in lists)
        
        for pos, neg, name, seed, cfg, step in itertools.islice(
            zip(
                itertools.cycle(poss), 
                itertools.cycle(negs), 
                itertools.cycle(names), 
                itertools.cycle(seeds or [None]), 
                itertools.cycle(cfgs), 
                itertools.cycle(steps)
            ), 
            max_len
        ):
            prompts.add(
                positive=pos, 
                negative=neg, 
                name=name, 
                seed=seed, 
                cfg=cfg, 
                steps=step
            )

    t2i = SDXLPainter(model)
    
    image = t2i.paint(prompts, grid=grid, annotate=annotate)
    
    del t2i
    gc.collect()
    torch.cuda.empty_cache()
    
