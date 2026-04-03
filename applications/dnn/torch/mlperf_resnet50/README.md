# Example inference script for MLPerf ResNet50-v1.5 on ImageNet classification
## Data
Prepare the ImageNet ILSVRC 2012 validation dataset (1000 classes). The required structure of the image directory is the same as what is used by [`torchvision.datasets.ImageNet`](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html).

First, download the validation dataset (`ILSVRC2012_img_val.tar`) and the development kit (`ILSVRC2012_devkit_t12.tar.gz`). Then, use the metadata in the development kit to organize the 50,000 validation images into 1000 directories, one for each class. These directories should be inside a directory called `val`.

In the inference scripts, `inference_resnet50.py` and `baseline_inference_resnet50.py`, set `ImageNet_root` to the path of the directory that contains `val`.

## PyTorch baseline inference

To run inference on ImageNet using ResNet50-v1.5 in PyTorch (no CrossSim), run:
```bash
python baseline_inference_resnet50.py
```

This can be used to test that you can properly load the ImageNet dataset and run inference using the ResNet50-v1.5 weights from the MLPerf Inference Benchmark.

The ImageNet ILSVRC2012 validation set contains 50,000 images. Inference can be restricted to a subset of the dataset by setting `Ntest`. The accuracy on the full validation set should be ~76.5%.

## Analog inference simulation

To simulate inference using ResNet50-v1.5 in CrossSim, run:
```bash
python inference_resnet50.py
```

This will simulate inference using CrossSim based on the analog hardware parameters that are set inside the script. 

For ResNet50-v1.5, the layer-wise calibrated input ranges and ADC ranges (for some common hardware parameters) are available. For an example of how to determine these calibrated ranges, see the calibration scripts in the CIFAR-10 directory.
