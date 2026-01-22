import re
import shutil
from pathlib import Path


import re
from pathlib import Path


def find_referenced_images(root):
    root = Path(root)
    text_exts = {".txt", ".md"}
    image_exts = r"png|jpg|jpeg|gif|webp|bmp|tiff|svg"

    # Captures:
    # 1. ![[Embeds]]
    # 2. [[Links.png]]
    # 3. (Markdown.png)
    # 4. "Quoted.png"
    # 5. YAML/Plain text lines ending in extension
    pattern = re.compile(
        rf"""(?xi)
        (?:
            !\[\[([^\]|]+)                      
            |\[\[([^\]|]+\.(?:{image_exts}))    
            |\(([^)]+\.(?:{image_exts}))\)      
            |["']([^"']+\.(?:{image_exts}))["'] 
            |(?<=:\s)([^\n]+?\.(?:{image_exts}))(?=\s|$)
        )
        """
    )

    images = set()

    for file in root.rglob("*"):
        if file.suffix.lower() in text_exts:
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
                for match in pattern.finditer(content):
                    # Extract the captured group that is not None
                    ref = next((g for g in match.groups() if g is not None), None)
                    if ref:
                        # Path(ref.strip()).name gets "image.png" from "path/to/image.png"
                        # and handles spaces/apostrophes correctly.
                        images.add(Path(ref.strip()).name)
            except Exception:
                continue

    return images


def find_all_images(root):
    root = Path(root)
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}

    name_to_paths = {}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            filename = p.name
            name_to_paths.setdefault(filename, []).append(p)

    return set(name_to_paths.keys()), name_to_paths


ROOT = Path("/home/c/Projects/Pen and Paper/Vascollan")
TARGET = Path("/home/c/Projects/Pen and Paper/Vascollan_Old_Stuff/Unused_Images")
TARGET.mkdir(parents=True, exist_ok=True)

ref = set(find_referenced_images(ROOT))
img_names, img_paths = find_all_images(ROOT)

unreferenced = img_names - ref

#print(unreferenced)
print(f"{len(unreferenced)}/{len(img_names)} images unreferenced")

for name in sorted(unreferenced):
    for src in img_paths[name]:
        dst = TARGET / src.name
        shutil.move(src, dst)
        print(f"Copied: {src} → {dst}")
