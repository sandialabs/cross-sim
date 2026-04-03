# Example inference script for CIFAR-10 image classification

## Inference simulation

This inference script can be used to simulate analog inference using CrossSim on one of four pre-trained ResNet models on the CIFAR-10 test set. The analog hardware parameters are set inside the `inference_cifar10_resnet.py` script. The ResNet model can be selected using the depth parameter `n`. The number of test images (up to 10,000) can be set using `N`.

For these ResNet models, the layer-wise calibrated input ranges and ADC ranges (for some common hardware parameters) are available.

To simulate inference, run:
```bash
python inference_cifar10_resnet.py
```

## Input range calibration

The script `calibrate_inputs.py` shows an example of a script that is used to calibrate the input ranges for a given model. The script first runs inference on a small number of images to profile the layer input data, then uses the profiled data to optimally calibrate each layer's input range. The calibrated ranges can then enable optimal inference accuracy with input quantization enabled.

This script profiles input data using 500 images from the CIFAR-10 training set. When profiling, turn off both input and ADC quantization. Then run:
```bash
python calibrate_inputs.py
```

The calibrated ranges will be saved to a file in `/calibrated_config`. Before running, check the path of the saved file to avoid unintentionally overwriting a file.

## ADC range calibration

The script `calibrate_adcs.py` shows an example of a script that is used to calibrate the ADC input ranges for a given model. As with input range calibraiton, this script profile ADC input data using 500 training images, then uses the profiled data to optimally calibrate each layer's ADC input range. The calibrated ranges can then enable optimal inference accuracy with ADC quantization enabled.

When profiling, turn off ADC quantization. Note that the calibrated ADC ranges can depend strongly on the other hardware parameters that are set, so check these carefully to ensure that they match your desired use case. When ready, run:
```bash
python calibrate_adcs.py
```

The calibrated ranges will be saved to a file in `/calibrated_config`. Before running, check the path of the saved file to avoid unintentionally overwriting a file.