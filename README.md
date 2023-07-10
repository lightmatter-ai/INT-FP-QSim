# INT-FP-QSim
INT-FP-QSim is a simulator that supports flexible evaluation of large language models (LLMs) and Vision Transformers for different numerical precisions, formats (integer or floating point) and their combinations. Please see [https://arxiv.org/abs/2307.03712](https://arxiv.org/abs/2307.03712) for further details on the simulator and the results obtained for different models. INT-FP-QSim is intended for research purposes only.

## Setup

### Prerequisites
INT-FP-QSim requires PyTorch.

As there are multiple installation options for PyTorch, we advise to follow [their installation directions](https://pytorch.org/get-started/locally/).
We also advise to use the `--index-url` option when installing PyTorch so the particular CUDA or CPU version is installed.

We have tested the repo using **PyTorch 1.13.0, CUDA 11.6 or CPU versions**.
This can be installed with: 
```pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html```

### Installation
INT-FP-QSim can be installed with `pip install -e .` from within the directory of the cloned repo.

### Quickstart: Running Resnet50 E4M3 quantization with the simulator
```
import torch
import torchvision
from int_fp_qsim.replace import replace_layers
from int_fp_qsim.formats import E4M3, BF16

model = torchvision.models.resnet50()
# Quantize inputs and weights to E4M3 and outputs in BF16
# Following function replaces layers of the model to support quantization
# Note that replace_layers is an in-place function
replace_layers(
    model,
    input_quantizer=E4M3,
    weight_quantizer=E4M3,
    output_quantizer=BF16
)
print(model)  # You can see the quantizer objects attached to layers

# Continue to evaluation
inputs = torch.randn(1, 3, 225, 225)
model.eval()
with torch.no_grad():
    model(inputs)
```

See `requirements.txt` for a full list of dependencies, including dependencies for running the examples provided. Example scripts for performing evaluation with different models is provided in the `examples` folder. See the corresponding `*.sh` files for the full commands.

## License
Licensed under Apache 2.0. Please see `LICENSE`.

## Disclaimer
This is not an officially supported Lightmatter product.

## Citing INT-FP-QSim
If you find this repository useful, please consider giving it a star and citation:
```
Nair, Lakshmi, et al. INT-FP-QSim: Mixed Precision and Formats For Large Language Models and Vision Transformers. arXiv, 7 July 2023. arXiv.org, https://doi.org/10.48550/arXiv.2307.03712.
```