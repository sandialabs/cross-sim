#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Parameterizable inference simulation script for MLPerf's ResNet50-v1.5
benchmark model on ImageNet. Inference is on the ILSVRC2012 validation set.
"""

import torch
import numpy as np
import warnings
import sys
import time
from build_resnet50 import resnet50
from dataset import create_ImageNet_dataloader

warnings.filterwarnings("ignore")
sys.path.append("../../")  # to import dnn_inference_params
sys.path.append("../../../../")  # to import simulator
from simulator.algorithms.dnn.torch.convert import (  # noqa:E402
    from_torch,
    convertible_modules,
    reinitialize,
)
from dnn_inference_params import dnn_inference_params  # noqa: E402
from find_adc_range import find_adc_range  # noqa:E402

useGPU = True  # use GPU?
Ntest = 50000  # number of images, up to 50000
batch_size = 32
Nruns = 1
print_progress = True

print("ImageNet: using " + ("GPU" if useGPU else "CPU"))
print(f"Number of images: {Ntest}")
print(f"Number of runs: {Nruns}")
print(f"Batch size: {batch_size}")
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

# ImageNet dataset location.
# Validation images must be sorted by label. This can be prepared by downloading
# the ILSVRC2012 validation set and sorting the images into unique directories
# by label using the metadata in the ILSVRC2012 dev kit.
ImageNet_root = "./imagenet/"
val_dataloader = create_ImageNet_dataloader(
    ImageNet_root, Ntest=Ntest, batch_size=batch_size
)

# Load Pytorch model
resnet50_model = resnet50(pretrained=True, ckpt_file="./resnet50_mlperf.pth").to(device)
resnet50_model.eval()
n_layers = len(convertible_modules(resnet50_model))

# Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    # Ideal mode means no analog hardware non-idealities are simulated
    "ideal": False,
    # Mapping style
    "core_style": "BALANCED",
    "Nslices": 1,
    # Weight value representation and precision
    "weight_bits": 8,
    "weight_percentile": 100,
    "digital_bias": True,
    # Memory device
    "Rmin": 1e4,
    "Rmax": 1e6,
    "infinite_on_off_ratio": False,
    "error_model": "none",
    "alpha_error": 0.00,
    "proportional_error": False,
    "noise_model": "none",
    "alpha_noise": 0.00,
    "proportional_noise": False,
    "drift_model": "none",
    "t_drift": 0,
    # Array properties
    "NrowsMax": 1152,  # inputs
    "NcolsMax": 256,  # outputs
    "Rp_row": 0,  # ohms
    "Rp_col": 0,  # ohms
    "interleaved_posneg": False,
    "subtract_current_in_xbar": True,
    "current_from_input": True,
    # Input quantization
    "input_bits": 8,
    "input_bitslicing": False,
    "input_slice_size": 1,
    # ADC
    "adc_bits": 8,
    "adc_range_option": "CALIBRATED",
    "adc_type": "generic",
    "adc_per_ibit": False,
    # Simulation parameters
    "useGPU": useGPU,
}

# Load input limits
input_ranges = np.load("./calibrated_config/dac_limits_ResNet50v15.npy")

# Load ADC limits
adc_ranges = find_adc_range(base_params_args, n_layers)

# Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k["positiveInputsOnly"] = False if k == 0 else True
    params_args_k["input_range"] = input_ranges[k]
    params_args_k["adc_range"] = adc_ranges[k]
    params_list[k] = dnn_inference_params(**params_args_k)

# Convert PyTorch layers to analog layers
analog_resnet50 = from_torch(
    resnet50_model, params_list, fuse_batchnorm=True, bias_rows=0
)
analog_resnet50.eval()

# Run inference and evaluate accuracy
accuracies = np.zeros(Nruns)
for m in range(Nruns):
    T1 = time.time()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = analog_resnet50(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if print_progress:
                print(
                    f"Image {total}/{Ntest}, top-1 accuracy so far = "
                    f"{100 * correct / total:.2f}%",
                    end="\r",
                )
    T2 = time.time()
    top1 = correct / total
    accuracies[m] = top1
    print("\nInference finished. Elapsed time: {:.3f} sec".format(T2 - T1))
    print(f"Top-1 accuracy: {100 * top1:.3f}% ({correct}/{total})\n")
    if m < (Nruns - 1):
        reinitialize(analog_resnet50)

if Nruns > 1:
    print("==========")
    print("Mean accuracy:  {:.3f}%".format(100 * np.mean(accuracies)))
    print("Stdev accuracy: {:.3f}%".format(100 * np.std(accuracies)))
