#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Baseline PyTorch inference script for MLPerf's ResNet50-v1.5
benchmark model on ImageNet, for comparison with CrossSim simulation
results. Inference is on the ILSVRC2012 validation set.
"""

import torch
import warnings
import time
from build_resnet50 import resnet50
from dataset import create_ImageNet_dataloader

warnings.filterwarnings("ignore")

Ntest = 50000  # number of images
useGPU = True  # use GPU?
batch_size = 32

print("ImageNet: using " + ("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(Ntest))
print("Batch size: {:d}".format(batch_size))
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

TA = time.time()
correct, total = 0, 0
with torch.no_grad():
    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = resnet50_model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

TB = time.time()
print("\n== Baseline Pytorch evaluation ==")
print("Inference finished. Elapsed time: {:.3f} sec".format(TB - TA))
print(f"Top-1 accuracy: {100.0 * correct / total:.3f}% ({correct}/{total})")
