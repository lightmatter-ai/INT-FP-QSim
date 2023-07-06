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
Code for evaluating Codegen models on OpenAI HumanEval.
"""

import argparse
import os
import re

import numpy as np
import torch
import transformers
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
from transformers import CodeGenForCausalLM, CodeGenTokenizerFast

from int_fp_qsim.formats import E4M3, FP16
from int_fp_qsim.replace import replace_layers

# Used to enable running the generated code on system
# Uncomment the following to allow running code on the system
#os.environ["HF_ALLOW_CODE_EVAL"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Attention implementations vary across versions
assert transformers.__version__ == "4.27.4"


def parse_args():
    parser = argparse.ArgumentParser(description="CodeGen Analysis")
    parser.add_argument(
        "--model-name-or-path", type=str, help="Path or name of Codegen model to use"
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        help="Path or name of the dataset (OpenAI HumanEval) to use",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility"
    )
    parser.add_argument("--do_FP8_eval", action="store_true", help="Evaluate in FP8")
    args = parser.parse_args()
    return args


def quantize_model(model):
    replace_layers(
        model, input_quantizer=E4M3, weight_quantizer=E4M3, output_quantizer=FP16
    )
    return model


def clean_test_input(test, entry_point):
    test += f"\ncheck({entry_point})"
    return test


def main():
    args = parse_args()
    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Create tokenizer and model
    tokenizer = CodeGenTokenizerFast.from_pretrained(args.model_name_or_path)
    model = CodeGenForCausalLM.from_pretrained(args.model_name_or_path)

    if args.do_FP8_eval:
        model = quantize_model(model)

    print(model)
    model.cuda()

    # Load the dataset and code evalutor
    human_eval = load_dataset(args.dataset_name_or_path)
    code_eval_metric = load("code_eval")

    # Perform inference, track metrics on test set
    avg_pass_at_1 = []
    model.eval()
    for i in tqdm(range(len(human_eval["test"]))):
        prompt = human_eval["test"][i]["prompt"]
        test = human_eval["test"][i]["test"]
        entry_point = human_eval["test"][i]["entry_point"]

        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            # Max length was determined based on max number of tokens in test set
            generated_ids = model.generate(
                input_ids.cuda(), max_length=428, pad_token_id=50256
            )
            candidates = [
                tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    truncate_before_pattern=[re.escape("<|endoftext|>")],
                )
            ]

            test = clean_test_input(test, entry_point)

            # Compute metric with generated code
            pass_at_k, _ = code_eval_metric.compute(
                references=[test], predictions=[candidates], k=[1]
            )
            print(f"Sample {i+1}: {pass_at_k['pass@1']}")
            avg_pass_at_1.append(pass_at_k["pass@1"])

    print(f"Avg. Pass@1:: {np.mean(avg_pass_at_1)}")


if __name__ == "__main__":
    main()
