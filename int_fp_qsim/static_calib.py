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

import numpy as np
import torch
from pytorch_quantization.calib import HistogramCalibrator
from pytorch_quantization.calib.histogram import (_compute_amax_mse,
                                                  _compute_amax_percentile)
from pytorch_quantization.utils import reduce_amax

from .replace import DEFAULT_MAPPING


class Calibrator:
    def __init__(self, name, num_bits, unsigned):
        self.name = name
        self.hc = HistogramCalibrator(num_bits, None, unsigned)
        self._tensor_max = -100.0

    def __call__(self, module, inputs, outputs):
        self.hc.collect(inputs[0])
        # Track maximum computed value
        self._tensor_max = max(self._tensor_max, torch.max(torch.abs(inputs[0])))

    def compute_amax(self, method, percentile):
        if method == "max":
            return self._tensor_max

        # Else compute via histograms
        return self.hc.compute_amax(method=method, percentile=percentile)


def _attach_histogram_calibrators(model, num_bits, unsigned):
    input_histograms = {}
    handles = []
    # Static calibration can be done on the replaced or unreplaced model.
    # Since layer types are different for both, using layer names
    # reconciles the difference.
    layer_type_names = [x.__name__ for x in DEFAULT_MAPPING]
    for name, module in model.named_modules():
        if hasattr(module, "weight") and type(module).__name__ in layer_type_names:
            # Initialize a calibrator for each module
            hc_inputs = Calibrator(name, num_bits, unsigned)
            input_histograms[name] = hc_inputs
            handles.append(module.register_forward_hook(hc_inputs))

    return input_histograms, handles


def _compute_and_attach_scales(model, histograms, method, percentile=99.99):
    for name, module in model.named_modules():
        if name in histograms:
            print(f"Calibrate input of {name}")
            amax_input = histograms[name].compute_amax(method, percentile)
            # Attach scale as attribute to the model
            module.amax_input = amax_input


def _remove_hooks(handles):
    for h in handles:
        h.remove()


def calibrate_inputs(
    model,
    num_bits,
    dataloader,
    unsigned=False,
    num_samples=5,
    batch_process_func=None,
    method="max",
    percentile=99.99,
):
    """Perform static calibration of input activations.

    Arguments:
        model (torch.nn.Module): The model to be calibrated.
        num_bits (int): Number of bits to use for quantization.
        dataloader (torch.utils.data.DataLoader): Pytorch dataloader for
            loading calibration data.
        unsigned (bool): Use unsigned or signed quantization.
        num_samples (int): Number of data batches to use for calibration.
            These many data batches are retrieved from the dataloader.
        batch_process_func (callable): A function that indicates how the
            model should process each batch of data. By default, data is
            passed as `model(data)` or `model(**data)`.
        method (str): Calibration method to use: "max", "entropy", "MSE",
            or "percentile".
        percentile (float): Value of percentile to use if "percentile"
            method specified for calibration.
    """
    # Attach histogram data collectors
    input_histograms, handles = _attach_histogram_calibrators(model, num_bits, unsigned)

    # Pass data through model for collection
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print(f"Processing data sample #{i+1} / {num_samples}")
            if batch_process_func:
                batch_process_func(model, data)
            else:
                try:
                    model(data)
                except:
                    model(**data)
            if i == num_samples - 1:
                break

    # Use the collected data histograms to compute the scales and add them
    # to the layers as attributes
    _compute_and_attach_scales(model, input_histograms, method, percentile)

    # Remove data collection handles at the end
    _remove_hooks(handles)


# Adapted from TensorRT Pytorch Quantization:
# https://github.com/NVIDIA/TensorRT with modifications.
def calibrate_weights(
    model,
    num_bits,
    unsigned,
    method="max",
    perchannel=False,
    percentile=99.99,
    num_bins=2048,
):
    """Perform static calibration of weights.

    Arguments:
        model (torch.nn.Module): The model to be calibrated.
        num_bits (int): Number of bits to use for quantization.
        unsigned (bool): Use unsigned or signed quantization.
        method (str): Calibration method to use: "max", "entropy", "MSE",
            or "percentile".
        perchannel (bool): Whether to calibrate on per-tensor (if False),
            or perchannel basis (if set to True).
        percentile (float): Value of percentile to use if "percentile"
            method specified for calibration.
        num_bins (int): Number of histogram bins to use for calibration.
    """
    # Static calibration can be done on the replaced or unreplaced model.
    # Since layer types are different for both, using layer names
    # reconciles the difference.
    layer_type_names = [x.__name__ for x in DEFAULT_MAPPING]
    for name, module in model.named_modules():
        if hasattr(module, "weight") and type(module).__name__ in layer_type_names:
            print(f"Calibrate weight of {name}")
            channel_second_modules = [
                "ConvTranspose1d",
                "ConvTranspose2d",
                "ConvTranspose3d",
            ]
            if perchannel:
                axis = 1 if type(module).__name__ in channel_second_modules else 0
            else:
                axis = None
            axis_size = module.weight.shape[axis] if axis is not None else 1

            # Histogram is always collected even if method is "max". Although "max" is supported here
            # but it is not the primary usage of this function
            if axis is None:
                calib_hist, calib_bin_edges = np.histogram(
                    module.weight.abs().cpu().detach().numpy(), bins=2048
                )
                calib_hist = [calib_hist]
                calib_bin_edges = [calib_bin_edges]
            else:
                calib_hist = []
                calib_bin_edges = []
                for i in range(axis_size):
                    hist, bin_edges = np.histogram(
                        module.weight.index_select(
                            axis, torch.tensor(i, device=module.weight.device)
                        )
                        .abs()
                        .cpu()
                        .detach()
                        .numpy(),
                        bins=num_bins,
                    )
                    calib_hist.append(hist)
                    calib_bin_edges.append(bin_edges)

            calib_amax = []
            if method == "max":
                if axis is not None:
                    reduce_axis = list(range(module.weight.dim()))
                    reduce_axis.remove(axis)
                    calib_amax.append(reduce_amax(module.weight, axis=reduce_axis))
                else:
                    calib_amax.append(torch.max(torch.abs(module.weight)))
            elif method == "mse":
                for i in range(axis_size):
                    calib_amax.append(
                        _compute_amax_mse(
                            calib_hist[i], calib_bin_edges[i], num_bits, unsigned
                        )
                    )
            elif method == "percentile":
                for i in range(axis_size):
                    calib_amax.append(
                        _compute_amax_percentile(
                            calib_hist[i], calib_bin_edges[i], percentile
                        )
                    )
            else:
                raise TypeError("Unsupported calibration method {}".format(method))

            if axis is None:
                calib_amax = calib_amax[0]
            else:
                calib_amax_shape = [1] * module.weight.dim()
                calib_amax_shape[axis] = module.weight.shape[axis]
                calib_amax = torch.stack(calib_amax).reshape(calib_amax_shape)
            module.amax_weight = calib_amax.detach().cpu().numpy()
