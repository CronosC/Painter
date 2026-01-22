import torch
import logging
from PIL import Image
from PIL import ImageDraw, ImageFont
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
from compel import Compel, ReturnedEmbeddingsType
from painter.prompt import PromptGrid
import os
import gc


import os
import torch
import logging
import gc
from abc import ABC, abstractmethod
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import difflib
import textwrap
from diffusers import StableDiffusionXLImg2ImgPipeline


class BasePainter(ABC):
    """Abstract base for all diffusion-based painters."""

    def __init__(self, model_name, dtype=torch.float16, output_dir="results"):
        self.dtype = dtype
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.pipeline = self._setup_pipeline(model_name)
        self._apply_optimizations()
        self._setup_scheduler()

    @abstractmethod
    def _setup_pipeline(self, model_name):
        """Load the specific pipeline type."""
        pass

    def _apply_optimizations(self):
        """Shared VRAM optimizations for 8GB cards."""
        self.pipeline.enable_model_cpu_offload()  # Better than .to("cuda") for OOM
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_attention_slicing("max")
        self.pipeline.enable_vae_slicing()

    def _setup_scheduler(self):
        """Default high-quality scheduler."""
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

    @abstractmethod
    def _run_inference(self, prompt, generator):
        """Model-specific call (e.g., prompt_embeds vs prompt)."""
        pass

    def _get_font(self, font_size):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf",
            "Tahoma.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        return ImageFont.load_default()

    def _get_str_diff(self, target_str, reference_words_set):
        target_words = target_str.split(", ")
        diff = [w for w in target_words if w not in reference_words_set]
        return " ".join(diff)

    def _draw_wrapped_text(self, draw, text, x, y, max_width, font, anchor="mm"):
        if not text:
            return

        avg_char_width = font.getlength("x") or 10
        # Reduce max_width slightly for internal padding
        chars_per_line = max(1, int((max_width - 20) / avg_char_width))
        lines = textwrap.wrap(text, width=chars_per_line)

        # join lines
        wrapped_text = "\n".join(lines)

        # Use spacing to ensure the lines don't touch
        draw.multiline_text(
            (x, y),
            wrapped_text,
            fill="white",
            font=font,
            anchor=anchor,
            align="center",
            spacing=6,
        )

    def _get_common_params(self, all_prompts):
        attributes = ["positive", "negative", "seed", "cfg", "steps"]
        common_parts = {}
        summary = []

        for attr in attributes:
            values = [getattr(p, attr) for p in all_prompts]
            if len(set(values)) == 1:
                # Fully shared
                summary.append(f"{attr.upper()}: {values[0]}")
                common_parts[attr] = set(str(values[0]).split(", "))
            elif attr in ["positive", "negative"]:
                # Partially shared text: Intersection of all word sets
                word_sets = [set(str(v).split(", ")) for v in values]
                common_words = set.intersection(*word_sets)
                if common_words:
                    # Keep original order for the common part display
                    ordered_common = [
                        w for w in str(values[0]).split(", ") if w in common_words
                    ]
                    summary.append(f"{attr.upper()}: {', '.join(ordered_common)}")
                    common_parts[attr] = common_words
                else:
                    common_parts[attr] = set()
            else:
                common_parts[attr] = set()

        return " \n ".join(summary), common_parts

    def _generate_dimension_label(
        self, subset_prompts, all_prompts, common_parts, prefix
    ):
        attributes = ["positive", "negative", "seed", "cfg", "steps"]
        unique_features = []
        ref_p = subset_prompts[0]

        for attr in attributes:
            all_vals = {getattr(p, attr) for p in all_prompts}
            subset_vals = {getattr(p, attr) for p in subset_prompts}

            # Dimension check: Varies in grid, fixed in this row/col
            if len(all_vals) > 1 and len(subset_vals) == 1:
                val = getattr(ref_p, attr)
                if attr in ["positive", "negative"]:
                    # Subtract the global common words from this dimension's fixed string
                    diff = self._get_str_diff(val, common_parts.get(attr, set()))
                    if diff:
                        unique_features.append(f"{attr}: {diff}")
                else:
                    unique_features.append(f"{attr}: {val}")

        return f"{prefix}\n" + ("\n".join(unique_features) if unique_features else "")

    def paint(
        self, prompts, prefix: str = "", grid: tuple[int, int] = (1, 1), annotate=True, overwrite=False
    ):
        rows, cols = grid
        per = rows * cols
        font_size, top_header_h, col_header_h, left_margin = 22, 220, 150, 320
        results, buf, grid_idx = [], [], 0
        font = self._get_font(font_size)

        for i, prompt in enumerate(prompts, start=1):
            generator = torch.Generator("cuda").manual_seed(prompt.seed)
            logging.info(f"Generating {prompt.width}x{prompt.height} image: {prompt.name}")
            image = self._run_inference(prompt, generator).convert("RGB")
            results.append(image)
            buf.append((image, prompt))

            if len(buf) == per:
                w, h = buf[0][0].size
                grid_prompts = [p for _, p in buf]

                if annotate:
                    total_w, total_h = (
                        cols * w + left_margin,
                        rows * h + top_header_h + col_header_h,
                    )
                    offset_x, offset_y = left_margin, top_header_h + col_header_h
                else:
                    total_w, total_h = cols * w, rows * h
                    offset_x, offset_y = 0, 0

                out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
                draw = ImageDraw.Draw(out)

                if annotate:
                    common_text, common_parts_dict = self._get_common_params(
                        grid_prompts
                    )
                    self._draw_wrapped_text(
                        draw,
                        f"\n{common_text}",
                        total_w // 2,
                        top_header_h // 2,
                        total_w - 60,
                        font,
                    )

                    prompts_2d = [
                        grid_prompts[r * cols : (r + 1) * cols] for r in range(rows)
                    ]

                    for r in range(rows):
                        label = self._generate_dimension_label(
                            prompts_2d[r], grid_prompts, common_parts_dict, f"r{r}: "
                        )
                        y_c = offset_y + (r * h) + (h // 2)
                        self._draw_wrapped_text(
                            draw, label, left_margin // 2, y_c, left_margin - 40, font
                        )

                    for c in range(cols):
                        col_subset = [prompts_2d[r][c] for r in range(rows)]
                        label = self._generate_dimension_label(
                            col_subset, grid_prompts, common_parts_dict, f"c{c}: "
                        )
                        x_c = offset_x + (c * w) + (w // 2)
                        y_c = top_header_h + (col_header_h // 2)
                        self._draw_wrapped_text(draw, label, x_c, y_c, w - 20, font)

                for j, (im, _) in enumerate(buf):
                    out.paste(
                        im, (offset_x + (j % cols) * w, offset_y + (j // cols) * h)
                    )
                    
                output_path = os.path.join(self.output_dir, f"{prefix}{prompt.name}.png")
                if not os.path.exists(output_path) or overwrite:
                    out.save(output_path)
                else:
                    print(f"Skipping {output_path}, file exists and overwrite=False")
                    
                grid_idx += 1
                buf.clear()

            torch.cuda.empty_cache()
            gc.collect()

        return results


class SD15Painter(BasePainter):
    """SD 1.5 Implementation."""

    def _setup_pipeline(self, model_name):
        return StableDiffusionPipeline.from_single_file(
            model_name, torch_dtype=self.dtype, safety_checker=None
        )

    def _run_inference(self, prompt, generator):

        return self.pipeline(
            prompt=prompt.positive,
            negative_prompt=prompt.negative,
            width=prompt.width,
            height=prompt.height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]


class SDXLPainter(BasePainter):
    """SDXL Implementation with Compel (Long Prompts)."""

    def _setup_pipeline(self, model_name):
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_name, torch_dtype=self.dtype
        )
        self.compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        return pipe

    def _run_inference(self, prompt, generator):
        p_embeds, p_pooled = self.compel(prompt.positive)
        n_embeds, n_pooled = self.compel(prompt.negative)

        cfg = prompt.cfg if prompt.cfg is not None else 7.0
        steps = prompt.steps if prompt.steps is not None else 30

        return self.pipeline(
            prompt_embeds=p_embeds,
            pooled_prompt_embeds=p_pooled,
            negative_prompt_embeds=n_embeds,
            negative_pooled_prompt_embeds=n_pooled,
            width=prompt.width,
            height=prompt.height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]


from diffusers import StableDiffusionXLInpaintPipeline


class SDXLInpaintPainter(BasePainter):
    """SDXL Inpainting/Masking Implementation."""

    def _setup_pipeline(self, model_name):
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            model_name, torch_dtype=self.dtype
        )
        self.compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        return pipe

    def _run_inference(self, prompt, generator):
        p_embeds, p_pooled = self.compel(prompt.positive)
        n_embeds, n_pooled = self.compel(prompt.negative)

        cfg = prompt.cfg if prompt.cfg is not None else 7.0
        steps = prompt.steps if prompt.steps is not None else 30
        strength = getattr(prompt, "strength", 0.75)

        return self.pipeline(
            prompt_embeds=p_embeds,
            pooled_prompt_embeds=p_pooled,
            negative_prompt_embeds=n_embeds,
            negative_pooled_prompt_embeds=n_pooled,
            image=prompt.image,
            mask_image=prompt.mask,
            width=prompt.width,
            height=prompt.height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
        ).images[0]


from diffusers import StableDiffusionXLImg2ImgPipeline


class SDXLImg2ImgPainter(BasePainter):
    """SDXL Image-to-Image Implementation."""

    def _setup_pipeline(self, model_name):
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_name, torch_dtype=self.dtype
        )
        self.compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        return pipe

    def _run_inference(self, prompt, generator):
        # Compel handles long prompts for XL
        p_embeds, p_pooled = self.compel(prompt.positive)
        n_embeds, n_pooled = self.compel(prompt.negative)

        cfg = prompt.cfg if prompt.cfg is not None else 7.0
        steps = prompt.steps if prompt.steps is not None else 30
        # strength: 1.0 is full noise, 0.0 is original image
        strength = getattr(prompt, "strength", 0.75)

        return self.pipeline(
            image=prompt.image,
            prompt_embeds=p_embeds,
            pooled_prompt_embeds=p_pooled,
            negative_prompt_embeds=n_embeds,
            negative_pooled_prompt_embeds=n_pooled,
            guidance_scale=cfg,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
        ).images[0]
