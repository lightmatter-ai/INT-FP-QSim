# INT-FP-QSim
INT-FP-QSim is a simulator that supports flexible evaluation of large language models (LLMs) and Vision Transformers for different numerical precisions, formats (integer or floating point) and their combinations. INT-FP-QSim leverages a subset of functions from existing open-source repos such as [TensorRT](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization), [QPytorch](https://github.com/Tiiiger/QPyTorch), and [AIMET](https://github.com/quic/aimet), to enable support for floating point formats and integer formats. INT-FP-QSim further extends the functions to support flexible floating point formats and static max calibration for input activations, while providing an easy interface for users to modify their models and run the simulation. 

INT-FP-QSim also provides the following:
1. Modified implementations of various attention, linear and convolutional layers that utilize the flexible quantization functions of this repo. 
2. Support for static calibration strategies with floating point representation.
3. Support for performing Quantization-aware-training for accuracy recovery with both integer and floating point formats.
4. Example scripts for running evaluation with Codegen (`eval_cg.py`), Maskformer (`eval_mf.py`), Stable Diffusion (`eval_sd.py`), and LLMs like OPT (`eval_opt.py`).

Please see [ADD LINK] for further details on the simulator and the results obtained for different models. Currently supported layers include: Linear, Conv1d, Conv2d, ConvTranspose2d, MultiheadAttention, OPTAttention, BERTSelfAttention, DetrAttention, MaskformerSwinSelfAttention, Attention (cross-attention for stable diffusion) and CodeGenAttention.

See `requirements.txt` for a full list of dependencies for the functions in this repo.

INT-FP-QSim is intended for research purposes only.

## Running simple FP8 quantization with the simulator
```
from replace import replace_layers
from format import E4M3, FP16

model = torchvision.models.resnet50()
# Quantize inputs and weights to E4M3 and outputs in FP16
# Following function replaces layers of the model to support quantization
# Note that replace_layers is an in-place function
replace_layers(
    model,
    input_quantizer=E4M3,
    weight_quantizer=E4M3,
    output_quantizer=FP16
)
print(model)  # You can see the quantizer objects attached to layers
# Continue to evaluation
```

**NOTE:** The quantizers specified as arguments to `replace_layers` is applied to all layers of the model, and different specifications for different layers is currently unsupported. 

**NOTE:** The quantizer functions perform "simulated" quantization which is a combination of quantize and dequantize operations.
See Eqn(8) of _FP8 Quantization: The Power of the Exponent_ [here](https://arxiv.org/pdf/2208.09225.pdf). Also described in Eqn(9) of _Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation_ [here](https://arxiv.org/pdf/2004.09602.pdf). Also see the simulator description in [ADD LINK].

Custom formats can be specified, e.g., you can use `QPytorch` to create the quantizer function for different exponent and mantissa bits (See `formats.py` for some examples). Note that the `input_quantizer`, `weight_quantizer` and `output_quantizer` arguments are callables. During forward pass, the inputs, weights and outputs of each layer in the model are passed through the corresponding quantizers to get the quantized tensors. A custom FP format can be specified as follows:

```
from qtorch import FloatingPoint
from qtorch.quant import quantizer

format_e3m4 = FloatingPointNumber(exp=3, man=4)
format_e5m2 = FloatingPointNumber(exp=5, man=2)

# A custom FP format that uses E3M4 in forward pass and E5M2 in backward
quant_func = quantizer(
    forward_num=format_e3m4,
    forward_rounding='nearest,
    backward_num=format_e5m2,
    backward_rounding='nearest'
)

replace_layers(
    model,
    input_quantizer=quant_func,
    weight_quantizer=quant_func,
    output_quantizer=quant_func 
)
```

## Static Quantization
**NOTE: This functionality is intended to be used with either `quantize_to_int()` or `quantize_to_fp()` functions in `quantizer.py`.**

The static quantization workflow is as follows:
1. Calibrate the inputs and weights of the FP32 model
2. Call `replace_layers` on the calibrated model
3. Proceed with the evaluation

The calibrators are adapted from [TensorRT](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) with several modifications, such as supporting static max calibration and custom batch process functions that explicitly specify how the model should process the calibration data. An example of how to perform step 1. is shown below for Resnet50. For OPT, `do_static_calibration()` function is provided in `eval_opt.py`. Note that in the example below, `INT8` and `INT4` use the `quantize_to_int()` function:
```
from torchvision.models import resnet50
from formats import INT8, INT4, FP16
from replace import replace_layers
from static_calib import calibrate_inputs, calibrate_weights

def do_static_calibration(
    model, dataloader, num_bits, method, percentile, num_samples=5
):
    # [OPTIONAL] Function that tells the model how to process the inputs.
    def batch_process_func(model, inputs):
        # Most commonly used for BERT / OPT
        model(inputs[0].cuda())

    # This function will pass `num_samples` batches of data
    # from `dataloader` and calibrate the model for `num_bits`
    # based on the specified `method`.
    calibrate_inputs(
        model.cuda(),
        num_bits,
        dataloader,
        method=method,
        batch_process_func=batch_process_func,
        percentile=percentile,
        num_samples=num_samples,

    )

    # This following function performs the same for the model.
    # No data batches are required here since this is done on weights.
    calibrate_weights(model, num_bits, False, method="max", perchannel=True)

model = resnet50(pretrained=True)
data_dir = "/data/datasets/imagenet"
data_loaders = get_dataloaders(
    data_dir,
    128,
    32,
    4,
)

# The following function will compute and attach static
# scales as attributes to the FP32 model layers.
do_static_calibration(
    model,
    data_loaders.train_loader,
    num_bits=8,
    method="percentile",
    percentile=99.99,
    num_samples=1
)

# Replace layers will copy the static scale attributes
# from the FP32 layers to the replaced quant layers.
replace_layers(
    model,
    input_quantizer=INT8,
    weight_quantizer=INT4,
    output_quantizer=FP16
)

# If you print the model at this stage, you should
# be able to see the computed scales attached as
# `amax_input` and `amax_weight`
print(model)

# During inference, each layer's forward call will
# check if scales are attached as layer attributes.
# If they are, then the model will use the attached
# scales to perform the quantization.
eval(model, data_loaders.val_loader, "cuda")
```

## Techniques for accuracy recovery
INT-FP-QSim provides support for Quantization-aware training (QAT) currently. Future iterations will include support for Adaptive Block Floating Point ([ABFP](https://arxiv.org/abs/2205.06287)).

### Quantization-aware Training (QAT)
**NOTE**: When running QAT, the backward pass of the quantize function _must_ be well-defined. For instance, see the backward pass of the quantizer [here](https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L309). This definition uses a PWL (piecewise-linear) implementation for the backward pass that works very well during QAT. Without the backward pass defined in this manner, the model will not train correctly. The `quantize_to_int` function already has the PWL backward pass since it is based on TensorRT, allowing users to directly perform QAT. However, custom user-specified functions would need to have a custom backward pass implementation.

To run QAT, simply apply the quantizers and train the model as usual:

```
from replace import replace_layers
from format import INT8, FP16

model = torchvision.models.resnet50()
# Quantize inputs and weights using INT8 and outputs in FP16
replace_layers(
    model,
    input_quantizer=INT8,
    weight_quantizer=INT8,
    output_quantizer=FP16 
)

# Proceed with training...
```

## License
Licensed under Apache 2.0. Please see `LICENSE`.

## Disclaimer
This is not an officially supported Lightmatter product.

## Citing INT-FP-QSim
If you find this repository useful, please consider giving it a star and citation:
```
[ADD CITATION FOR PAPER]
```