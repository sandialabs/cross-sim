# CrossSim

CrossSim is a Python-based simulator of analog matrix vector multiplication (MVM) accelerators designed to help answer questions about how analog hardware effects impact solution quality. CrossSim can model a wide range of analog MVM arrays with different architectural choices including array size, the representation of negative numbers, the representation of multi-bit inputs, and bit sliced weights. Additionally, CrossSim can model device and circuit non-idealities including programming errors, cycle-to-cycle read noise, precision loss in analog-to-digital conversion, and parasitic metal resistances. CrossSim does not explicitly model the energy, area, or speed of analog MVM accelerators. A full description of CrossSim's parameters and modeling assumptions can be found in the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf).

## Requirements
CrossSim has been tested on Ubuntu 18.04, and Windows 10 using Python 3.7.6.

CrossSim requires the following Python packages:
* TensorFlow 2.4.1 (includes Keras 2.4.0)
* Numpy 1.20.3
* SciPy 1.7.1
* Pandas 1.3.3
* MatPlotLib 3.4.3
* CuPy 8.3.0 with CUDA 10.2, or Cupy 10.3.1 with CUDA 11.2 (if GPU acceleration is enabled)

Several of the neural network models provided with CrossSim may require additional packages.
ImageNet models may require:
* OpenCV 4.5.4
* Torchvision 0.11.1

Inference simulation of Larq quantized models requires:
* Larq 0.12.1

CrossSim has been tested with the version numbers above, but other versions may work as well.

## Getting Started
CrossSim uses git submodules to distribute [neural network datasets and device lookup tables for training](https://github.com/sandialabs/cross-sim-data), and [neural network models](https://github.com/sandialabs/cross-sim-models). After cloning this repository, the following commands will fetch the submodules.
```
git submodule init
git submodule update --progress
```

This command downloads about 1.2GB of data for the two repositories (combined), which is then de-compressed. After cloning the submodules, you can test CrossSim Inference by running the following commands:
```
cd inference
python run_inference.py
```
This will run a CrossSim Inference simulation using a simple CNN and the MNIST dataset by default. The use of the run_inference.py script is explained in more detail in Chapter 3 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf).

Due to file size and copyright, this distribution does not include the ImageNet dataset. Section 4.3 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf) explains how to add your own copy of the ImageNet test set to CrossSim.

## Using CrossSim
CrossSim is primarily focused on neural network inference. Example scripts to perform inference are in the ``inference`` directory and described in Chapter 3 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf). If you are interested in benchmarking large neural networks, a CUDA-capable GPU is recommended.

If you would like to use the CrossSim array models for other analog MVM applications, the hardware models are located in the ``cross_sim/cross_sim/xbar_simulator`` directory of the repository.

## Adding New Device Models for Inference
Methods for applying device-specific models for programming errors, cycle-to-cycle read noise, and conductance drift are found in the corresponding files located inside the following directory:
```
/cross_sim/cross_sim/xbar_simulator/parameters/custom_device/
```
To implement a new device model for use in inference simulations, a new device model can be added as an option in these files. Please see Chapters 7.2-7.4 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf) for more details.

## Adding New Models, Datasets and Devices for Training
To use your own pretrained models and datasets for use with CrossSim Inference, or your own device experimental data for use with CrossSim Training, modify the url field in the .gitmodules folder. If you have no yet initialized the submodules, use the commands above. To change the submodule url after the submodules have been initialized, use the following commands:
```
git submodule sync
git submodule update --init --remote
```
Instructions for adding new neural network models and devices into CrossSim for use during inference can be found in Chapters 5 and 7 of [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf) respectively.

If you would like to contribute your device data or models, create a pull request against this repository and the [data](https://github.com/sandialabs/cross-sim-data) or [pre-trained models](https://github.com/sandialabs/cross-sim-models) repositories.

## Citing CrossSim
If you use CrossSim for research please cite:
```
@article{crossim,
  author={T. Patrick Xiao, Christopher H. Bennett, Ben Feinberg, Matthew J. Marinella, Sapan Agarwal},
  title={CrossSim 2.0 Inference Manual},
  doi = {10.2172/1869509},
  url = {https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual.pdf},
}
```