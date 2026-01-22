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
}

style = "classical medieval oil painting, rough oil brushstrokes, expressive painterly strokes, impasto highlights, painterly, chiaroscuro lighting, waist-up portrait"

os.makedirs(OUT_FOLDER, exist_ok=True)


def multi_replace(text, replacements):
    # Sort keys by length descending to ensure longest match is found first
    pattern = re.compile(
        "|".join(
            re.escape(k) for k in sorted(replacements.keys(), key=len, reverse=True)
        )
    )
    return pattern.sub(lambda m: replacements[m.group(0)], text)


# Process files
summarizer = LocalSummarizer(MODEL_NAME)
for filename in os.listdir(FOLDER):
    input_path = os.path.join(FOLDER, filename)

    if os.path.isfile(input_path):
        print(f"Processing: {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = filename + "\n" + content
        content = multi_replace(content, fixes)
        
        prompt_text = ""
        while not prompt_text.startswith("Medieval oil portrait of"):
            prompt_text = summarizer.summarize(
                content, system_prompt=SYSTEM_PROMPT_FILE, lang="de"
            )

            if ":" in prompt_text:
                prompt_text = prompt_text.split(":")[-1]
            prompt_text = prompt_text.replace('"', "")
            prompt_text = prompt_text.strip(" \n")

        prompt_text = style + ", " + prompt_text

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUT_FOLDER, f"{base_name}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

print("Done.")
