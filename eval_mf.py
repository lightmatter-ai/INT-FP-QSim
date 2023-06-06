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
Code for evaluating Maskformer on ADE20k.
"""

import argparse
import json
import os

import evaluate
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_url
from mit_semseg.lib.nn import user_scattered_collate
from mit_semseg.utils import intersectionAndUnion
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

from simulator.formats import E4M3, FP16
from simulator.replace import replace_layers

# create an argument parser that takes in the number of epochs to train, batch size, and learning rate
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of pretrained model to use")
parser.add_argument("--do_FP8_eval", action="store_true", help="evaluate in FP8")
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="batch size (N.B. only 1 sample is going into the model at a time)",
)
args = parser.parse_args()

for k, v in args.__dict__.items():
    print("{}: {}".format(k, v))

# the layer in the model that controls the number of classes to predict
replacer_layer_idx = 629

# see if cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_name = args.model_name
model = MaskFormerForInstanceSegmentation.from_pretrained(pretrained_model_name)
print(model.config)

root_dir = "/data/datasets/ADE20k"
train_odgt = "/data/datasets/ADE20k/ADEChallengeData2016/training.odgt"
test_odgt = "/data/datasets/ADE20k/ADEChallengeData2016/validation.odgt"
# https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"  # "ade20k-hf-doc-builder.json"
id2label = json.load(
    open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = num_classes = len(id2label)

feature_extractor = AutoImageProcessor.from_pretrained(
    pretrained_model_name, ignore_index=-1, do_reduce_labels=True
)

img_size = 640


def pixel_acc(preds, label):
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc


# create an ade20k dataset class
class ade20k(Dataset):
    def __init__(
        self, root_dir, odgt, feature_extractor=None, transform=None, **kwargs
    ):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.odgt = odgt
        self.parse_input_list(odgt, **kwargs)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, "r")]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print("# samples: {}".format(self.num_sample))

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, idx):
        img_path = self.list_sample[idx]["fpath_img"]
        img_path = os.path.join(self.root_dir, img_path)
        # make sure the image is in RGB format
        image = Image.open(img_path).convert("RGB")

        label_path = self.list_sample[idx]["fpath_segm"]
        label_path = os.path.join(self.root_dir, label_path)
        # make sure the image is in RGB format
        label = Image.open(label_path)

        if self.feature_extractor:
            sample = self.feature_extractor(image)
            X = sample["pixel_values"][0]
            X = torch.from_numpy(X)
            X = X[None, ...]
            sample["pixel_values"] = X

        return sample, label


model = model.to(device)

if args.do_FP8_eval:
    replace_layers(
        model, input_quantizer=E4M3, weight_quantizer=E4M3, output_quantizer=FP16
    )

print(model)
model = model.eval()

layers_req_grad = 0
tot_layers = 0

params_req_grad = 0
tot_params = 0

for param in model.named_parameters():
    if param[1].requires_grad:
        layers_req_grad += 1
        params_req_grad += param[1].nelement()
    tot_layers += 1
    tot_params += param[1].nelement()

print(
    "{0:,} layers require gradients (unfrozen) out of {1:,} layers".format(
        layers_req_grad, tot_layers
    )
)
print(
    "{0:,} parameters require gradients (unfrozen) out of {1:,} parameters".format(
        params_req_grad, tot_params
    )
)

train_dataset = ade20k(root_dir, train_odgt, feature_extractor=feature_extractor)
test_dataset = ade20k(root_dir, test_odgt, feature_extractor=feature_extractor)

# create a dataloader for the test dataset
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=user_scattered_collate,
    num_workers=4,
)

crit = nn.NLLLoss(ignore_index=-1)
metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        pred_labels, labels = eval_pred
        pred_labels = pred_labels.detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        # add a dimension to the labels and pred_labels
        labels = labels[None, ...]
        pred_labels = pred_labels[None, ...]

        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=-1,
            reduce_labels=True,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics


mx_count = 1000000
acc = []
miou = []
over_all_acc = []

mit_acc_list = []
mit_miou_list = []

L = []
I = []
U = []
for idx, batch in enumerate(test_loader):
    # for each sample in the batch
    print(f"Batch #{idx+1} / {len(test_loader)}")
    for sample in batch:
        # get the inputs and labels
        inputs = sample[0]
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        label = sample[1]
        # map the label to tensor
        # For some reason, the feature extractor is not subtracting the label by 1.
        label = torch.from_numpy(np.array(label)).to(torch.long)  # - 1
        seg_size = label.shape
        label = label[None, ...].to(device)
        # pass the inputs to the model
        with torch.no_grad():
            outputs = model(**inputs)
        # # get the logits
        predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(
            outputs, target_sizes=[seg_size]
        )[0]
        label = label - 1

        mit_seg_acc = pixel_acc(predicted_semantic_map, label)
        mit_acc_list.append(mit_seg_acc.item())

        predicted_semantic_map = predicted_semantic_map.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        intersection, union = intersectionAndUnion(
            predicted_semantic_map.squeeze(), label.squeeze(), num_classes
        )
        I.append(intersection)
        U.append(union)
        mit_iou = intersection.sum() / (union.sum() + 1e-10)
        mit_miou_list.append(mit_iou)
        print("mit_seg_acc: {:0.4f}\t mit_iou {:0.4f}".format(mit_seg_acc, mit_iou))

    if idx > mx_count:
        break

# make I and U to numpy array
I = np.array(I)
U = np.array(U)
# sum the I and U
I = I.sum(axis=0)
U = U.sum(axis=0)
mIoU_by_class = I / (U + 1e-10)

for i, _iou in enumerate(mIoU_by_class):
    print("class [{}], IoU: {:.4f}".format(i, _iou))

# compute average miou over class
avg_miou = mIoU_by_class.mean()
mit_acc_list = np.array(mit_acc_list)
print(
    "MIT: mit_mean_acc: {:0.4f}\tmit_mean_iou: {:0.4f}".format(
        mit_acc_list.mean(), avg_miou
    )
)
