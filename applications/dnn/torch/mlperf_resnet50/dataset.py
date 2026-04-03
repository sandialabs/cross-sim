#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import torch.utils.data
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.datasets as datasets
from pathlib import Path


class IncrementLabels(torch.utils.data.Dataset):
    """Increment the labels in the ImageNet dataset by 1.
    This is needed for the MLPerf ResNet50-v1.5 model which assumes
    1001 classes. Class 0 is a "none" class.
    """

    def __init__(self, dataset):
        """Initialize IncrementLables from a dataset."""
        self.dataset = dataset

    def __getitem__(self, index):
        """Returns dataset item incremented by 1."""
        image, label = self.dataset[index]
        return image, label + 1

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)


def shuffle_truncate(dataset, Ntest, seed=21):
    """Shuffle then truncate a dataset given a seed."""
    n = len(dataset)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    keep = perm[:Ntest]
    return Subset(dataset, keep)


def create_ImageNet_dataloader(ImageNet_root, Ntest=50000, batch_size=32, seed=21):
    """Create an ImageNet dataloader for inference."""
    # Set ImageNet dataset location here
    valdir = Path(ImageNet_root) / "val"
    if not Path(valdir).is_dir():
        raise FileNotFoundError(f"ImageNet val dataset not found under {valdir}")

    # MLPerf ResNet50-v1.5 requires zero-centered pixel values with
    # a range of 255.
    # Torchvision's InterpolationMode.BICUBIC is closest to the
    # cv2.INTER_AREA interpolator used by MLPerf.
    val_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.68 / 255.0, 116.78 / 255.0, 103.94 / 255.0],
            std=[1 / 255.0, 1 / 255.0, 1 / 255.0],
        ),
    ])

    val_dataset = datasets.ImageFolder(valdir, val_transforms)

    # ResNet50-v1.5 requires labels to be incremented by 1.
    val_dataset = IncrementLabels(val_dataset)

    # Shuffle and truncate the dataset
    # For deterministic results, set the seed.
    # Even if not truncating, shuffle ensures intermediate accuracy is
    # representative of the final accuracy.
    val_dataset = shuffle_truncate(val_dataset, Ntest, seed=seed)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return val_loader
