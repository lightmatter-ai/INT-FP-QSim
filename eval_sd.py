#!/usr/bin/env python
# Copyright 2023 Lightmatter, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for evaluating Stable Diffusion model on Conceptual Captions.
"""

import argparse
import csv
import os
from functools import partial

import diffusers
import numpy as np
import torch
import transformers
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from transformers import CLIPTokenizer

from simulator.formats import E4M3, FP16
from simulator.replace import replace_layers

# Attention implementations vary across versions
assert transformers.__version__ == "4.27.4"
assert diffusers.__version__ == "0.16.1"


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Analysis")
    parser.add_argument(
        "--model-name-or-path", type=str, help="Path or name of diffusion model to use"
    )
    parser.add_argument(
        "--clip-model-name-or-path",
        type=str,
        help="Path or name of the model to use for computing CLIP score",
    )
    parser.add_argument(
        "--dataset-name-or-path", type=str, help="Path or name of the dataset to use"
    )
    parser.add_argument(
        "--pipeline-module-parts",
        nargs="+",
        help="Parts of the pipeline that are torch modules",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility"
    )
    parser.add_argument("--do_FP8_eval", action="store_true", help="Evaluate in FP8")
    parser.add_argument(
        "--num-images-to-save",
        type=int,
        default=0,
        help="Number of generated images to save",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_output",
        help="Directory for saving images",
    )
    args = parser.parse_args()
    return args


def save_outputs(image, fname):
    # Save a single image as output
    image_cleaned = image.squeeze()
    image_cleaned = (image_cleaned * 255).astype("uint8")
    img = Image.fromarray(image_cleaned, "RGB")
    img.save(fname)


def check_token_length(prompt, tokenizer):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=False,
        return_tensors="pt",
    )

    tokens = text_input.input_ids.shape[-1]
    return tokens


def main():
    args = parse_args()

    # Create output directories if needed
    if args.num_images_to_save > 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    stable_diffusion_modules = ["vae", "text_encoder", "unet", "safety_checker"]
    sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(
        "cuda"
    )

    if args.do_FP8_eval:
        for part in stable_diffusion_modules:
            print(f"FP8 conversion of {part}...")
            module = getattr(sd_pipeline, part)
            if not hasattr(module, "named_modules"):
                raise ValueError(f"{part} does not have torch modules")
            replace_layers(
                module,
                input_quantizer=E4M3,
                weight_quantizer=E4M3,
                output_quantizer=FP16,
            )
            print(module)

    clip_score_fn = partial(clip_score, model_name_or_path=args.clip_model_name_or_path)
    # Tokenizer to test input length
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name_or_path)

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(
            torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
        ).detach()
        return round(float(clip_score), 4)

    prompts = load_dataset(args.dataset_name_or_path, split="validation")
    full_prompts = [prompts[i]["caption"] for i in range(1000)]

    generator = torch.manual_seed(args.seed)
    clip_scores = []
    prompt_img_mapping = {}
    for i, p in enumerate(full_prompts):
        # Check if prompts match desired length.
        # We discard prompts that are longer than the maximum
        # supported token length of the CLIP model.
        token_length = check_token_length(p, clip_tokenizer)
        if token_length > 77:
            continue

        images = sd_pipeline(
            p, num_images_per_prompt=1, generator=generator, output_type="numpy"
        ).images

        if i < args.num_images_to_save:
            fname = f"{args.output_dir}/image_{i}.png"
            save_outputs(images, fname)
            prompt_img_mapping[fname] = p

        sd_clip_score = calculate_clip_score(images, p)
        print(f"Image {i} score: {sd_clip_score}")
        clip_scores.append(sd_clip_score)

    print(f"Avg. CLIP score: {np.mean(clip_scores)}")
    if args.num_images_to_save > 0:
        fname = f"{args.output_dir}/prompts_for_saved_images.txt"
        with open(fname, "w") as f:
            reader = csv.writer(f)
            for img_file, prompt in prompt_img_mapping.items():
                reader.writerow((img_file, prompt))
        f.close()


if __name__ == "__main__":
    main()
