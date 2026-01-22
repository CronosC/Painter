import os
import gc
import torch
import re
import requests
from painter.summarizer import LocalSummarizer

FOLDER = "/home/c/Projects/Pen and Paper/Vascollan/05 Soziales/Personen"
SYSTEM_PROMPT_FILE = "prompts/system/portrait"
OUT_FOLDER = "prompts/automatic"
MODEL_NAME = "llama3"

fixes = {
    "Mittelreich": "zentraleuropäisch aussehend",
    "Khün": "mongolisch aussehenden Ureinwohnern",
    "Tulamidisch": "arabisch aussehend",
    "Geweihter": "Priester",
    "Elf": "Elfisch",
    "Badok": "in Menschlicher Kultur Aufgewachsen",
    "Horasier": "Mediterranes Aussehen",
}

# style = "classical medieval (oil painting:2), (rough oil brushstrokes:2), expressive (painterly:3) strokes, impasto highlights, painterly, chiaroscuro lighting, waist-up portrait"
style = "classical medieval (oil painting:2), (rough oil brushstrokes:2), expressive (painterly:2) strokes, shoulders-up portrait"
neg_style = "photorealistic, nudity, nsfw, explicit, bad quality, worst quality, worst detail, sketch, text, signature, christian symbols, christianity, modern accesoires, modern looking glasses, modern styles"

os.makedirs(OUT_FOLDER, exist_ok=True)


def multi_replace(text, replacements):
    # Sort keys by length descending to ensure longest match is found first
    pattern = re.compile(
        "|".join(
            re.escape(k) for k in sorted(replacements.keys(), key=len, reverse=True)
        )
    )
    return pattern.sub(lambda m: replacements[m.group(0)], text)

summarizer = LocalSummarizer(MODEL_NAME)

# Process files
for filename in os.listdir(FOLDER):
    input_path = os.path.join(FOLDER, filename)

    if os.path.isfile(input_path):
        print(f"Processing: {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = filename + "\n" + content
        og_content = content
        content = multi_replace(content, fixes)

        prompt_text = ""
        neg_text = ""
        
        
        if "01 Data/Bilder/Portraits/Anonyme Person.png" in content:
            while True:
                prompt_text = summarizer.summarize(
                    content, system_prompt=SYSTEM_PROMPT_FILE, lang="de"
                )
                prompt_text = prompt_text.replace("NEGATIVE", "").replace("POSITIVE", "")
                

                if ":" in prompt_text:
                    prompt_text = prompt_text.replace("\n", "")
                    splits = prompt_text.split(":")
                    if len(splits) != 4:
                        print("Incorrect output structure, trying again..")
                        continue
                    else:
                        prompt_text = splits[-2]
                        neg_text = splits[-1]
                        prompt_text = prompt_text.replace('"', "")
                        prompt_text = prompt_text.strip(" \n")
                        neg_text = neg_text.replace('"', "")
                        neg_text = neg_text.strip(" \n")
                        break
            addon = ""
            addon_neg = ""        
            if "  - Ork" in og_content:
                addon = addon + "black skin and very furry face and body, black fur on body, "
                addon_neg = addon_neg + "green skin, red skin, naked skin, "
            if "  - Khün" in og_content:
                addon = addon + "mongolian ethnicity, "
                addon_neg = addon_neg + "intricate clothing, metal armor, "

            prompt_text = style + ", " + addon + prompt_text
            neg_text = neg_style + ", " + addon_neg + neg_text

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(OUT_FOLDER, f"{base_name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
                
            output_path_neg = os.path.join(OUT_FOLDER, f"NEG {base_name}.txt")
            with open(output_path_neg, "w", encoding="utf-8") as f:
                f.write(neg_text)

print("Done.")
