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
Code for evaluating ImageBind on ImageNet dataset. Please use in conjunction
with the original ImageBind repo cloned from: https://github.com/facebookresearch/ImageBind
"""

import argparse
import copy
import json
import os
import time

# The "models" module is from: https://github.com/facebookresearch/ImageBind
import models.transformer as ib
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from huggingface_hub import cached_download, hf_hub_url
from models import imagebind_model
from models.imagebind_model import ModalityType
from models.multimodal_preprocessors import SimpleTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import int_fp_qsim.attention as qattn
import int_fp_qsim.layers as qnn
from int_fp_qsim.formats import E4M3, FP16
from int_fp_qsim.replace import DEFAULT_MAPPING, replace_layers


def get_parser():
    parser = argparse.ArgumentParser(
        description=("Evaluate ImageBind on ImageNet dataset.")
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        required=False,
        default="",
        help="Location of ImageBind checkpoint to be loaded.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        required=False,
        default=32,
        help="Validation data loader batch size",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to directory containing validation/test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--do-FP8-eval",
        action="store_true",
        help="Simulate model in FP8.",
    )
    return parser


def load_and_transform_text(text, device):
    # Obtain from https://github.com/facebookresearch/ImageBind
    BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def get_dataloaders(data_dir, eval_batch_size, num_workers=4):
    val_loader = None
    if eval_batch_size is not None:
        val_dir = os.path.join(data_dir, "val")
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        val_dataset = datasets.ImageFolder(
            val_dir,
            data_transform,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return val_loader


def quantize_model(model):
    mapping = copy.deepcopy(DEFAULT_MAPPING)
    additional_layers = {
        ib.MultiheadAttention: qattn.ImageBindMha,
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear: qnn.Linear,
    }
    mapping.update(additional_layers)
    replace_layers(
        model,
        input_quantizer=E4M3,
        weight_quantizer=E4M3,
        output_quantizer=FP16,
        mapping=mapping,
    )
    return model


def eval(model, data_loader, labels, device):
    print("Running evaluation.")
    model.eval()
    correct = 0
    start = time.time()
    with torch.no_grad():
        for img, target in tqdm(data_loader):
            img, target = img.to(device), target.to(device)
            inputs = {
                ModalityType.TEXT: load_and_transform_text(labels, device),
                ModalityType.VISION: img,
            }
            embeddings = model(inputs)
            output = torch.softmax(
                embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T,
                dim=-1,
            )
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    elapsed = int(time.time() - start)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    print(
        f"accuracy:      {accuracy:.01f}" + "\n"
        f"elapsed time:  {elapsed:d} sec." + "\n"
    )
    print("Evaluation complete.")


parser = get_parser()
args = parser.parse_args()
device = "cpu" if not torch.cuda.is_available() else torch.device("cuda")

manual_seed = args.seed
if manual_seed is None:
    print("--seed is None. Generating random seed.")
    manual_seed = torch.randint(99, (1,)).item()

torch.manual_seed(manual_seed)
print(f"Set torch random seed to: {manual_seed}")

if not args.load_checkpoint:
    # Get checkpoint from https://github.com/facebookresearch/ImageBind/tree/main
    model = imagebind_model.imagebind_huge(pretrained=True)
else:
    model = imagebind_model.imagebind_huge(pretrained=False)
    ckpt = torch.load(args.load_checkpoint)
    model.load_state_dict(ckpt)

model.to(device)

# Load validation dataset
val_loader = get_dataloaders(args.data_dir, args.eval_batch_size)

# Load mapping from class idx to text (human-readable class names)
repo_id = "huggingface/label-files"
filename = "imagenet-1k-id2label.json"
id2label = json.load(
    open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")
)
id2label = {int(idx): label for idx, label in id2label.items()}
labels = list(id2label.values())

if args.do_FP8_eval:
    print("Performing FP8 evaluation...")
    model = quantize_model(model)

# Perform evaluation
eval(model, val_loader, labels, device)
