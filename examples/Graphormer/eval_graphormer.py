from datasets import load_dataset
from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse

from int_fp_qsim.replace import replace_layers
from int_fp_qsim.formats import E4M3, FP16


def parse_args():
    parser = argparse.ArgumentParser(description='Graphormer Evaluation')
    parser.add_argument('--model-name-or-path', type=str, help='Path or name of graphormer checkpoint')
    parser.add_argument('--dataset-name-or-path', type=str, help='Path or name of the dataset to use')
    parser.add_argument('--eval-in-fp8', action='store_true', help='Evaluate in FP8')
    args = parser.parse_args()
    return args


def quantize_model(model):
    replace_layers(
        model,
        input_quantizer=E4M3,
        weight_quantizer=E4M3,
        output_quantizer=FP16
    )
    return model


def main():
    args = parse_args()
    # There is only one split on the hub
    dataset = load_dataset(args.dataset_name_or_path)
    dataset = dataset.shuffle(seed=0)
    dataset_processed = dataset.map(preprocess_item, batched=False)

    model = GraphormerForGraphClassification.from_pretrained(
        args.model_name_or_path,
        num_classes=2, # num_classes for the downstream task 
        return_dict=True
    )

    if args.eval_in_fp8:
        print("Performing FP8 evaluation...")
        model = quantize_model(model)

    print(model)
    model.cuda()

    dataloader = DataLoader(
        dataset_processed["validation"],
        batch_size=1,
        shuffle=False,
        collate_fn=GraphormerDataCollator()
    )

    correct = 0
    for data in enumerate(tqdm(dataloader)):
        inputs = {
            'input_nodes': data[1]['input_nodes'].cuda(),
            'input_edges': data[1]['input_edges'].cuda(),
            'attn_bias': data[1]['attn_bias'].cuda(),
            'in_degree': data[1]['in_degree'].cuda(),
            'out_degree': data[1]['out_degree'].cuda(),
            'spatial_pos': data[1]['spatial_pos'].cuda(),
            'attn_edge_type': data[1]['attn_edge_type'].cuda()
        }
        labels = data[1]['labels'].cuda()
        y = model(**inputs)
        prediction = torch.argmax(torch.round(torch.softmax(y['logits'], dim=1)))
        correct += int(prediction == labels)

    print(f"Accuracy: {correct * 100 / len(dataloader)}")


if __name__ == '__main__':
    main()
