# CrossSim

CrossSim is a GPU-accelerated, Python-based crossbar simulator designed to model analog in-memory computing for any application that relies on matrix operations: neural networks, signal processing, solving linear systems, and many more. It is an accuracy simulator and co-design tool that was developed to address how analog hardware effects in resistive crossbars impact the quality of the algorithm solution.

CrossSim has a Numpy-like API that allows different algorithms to be built on resitive memory array building blocks. CrossSim cores can be used as drop-in replacements for Numpy in application code to emulate deployment on analog hardware. It also has a special interface to model analog accelerators for neural network inference.

CrossSim can model device and circuit non-idealities such as arbitrary programming errors, conductance drift, cycle-to-cycle read noise, and precision loss in analog-to-digital conversion (ADC). It also uses a fast, internal circuit simulator to model the effect of parasitic metal resistances on accuracy. For neural network inference, it can simulate accelerators with significant parameterizability at the system architecture level and can be used to explore how design choices such as weight bit slicing, negative number representation scheme, ADC ranges, and array size affect the sensitivity to these analog errors. CrossSim can be accelerated on CUDA GPUs, and inference simulations have been run on large-scale, industry-standard deep neural networks such as ResNet50 on ImageNet. CrossSim's simulation speed on ResNet50 is within ~3X of TensorFlow-Keras using baseline (simplest) analog hardware simulation settings.

 __To simulate neural network training, please use [CrossSim Version 2.0](https://github.com/sandialabs/cross-sim/releases/tag/v2.0). Support for neural network training will be brought back in a future software update.__

CrossSim does not explicitly model the energy, area, or speed of analog accelerators. 

## Requirements
CrossSim has been tested on Ubuntu 18.04, and Windows 10 using Python 3.10.8.

CrossSim requires the following Python packages:
* Numpy 1.24.3
* SciPy 1.11.1
* TensorFlow 2.13.0 (for DNN inference)
* IPython 8.8.0 (for tutorials)
* MatPlotLib 3.7.2 (for tutorials and application examples)
* CuPy 8.3.0 with CUDA 10.2, or Cupy 12.1.0 with CUDA 11.1 (for GPU acceleration)

Several of the neural network models provided with CrossSim may require additional packages.
ImageNet models may require:
* OpenCV 4.5.4
* Torchvision 0.11.1

Inference simulation of Larq quantized models requires:
* Larq 0.12.1

CrossSim has been tested with the version numbers above, but other versions may work as well.

## Installation

CrossSim can be installed using pip using the following command from the base repository directory.
```
pip install .
```
If you plan to modify the internal internal CrossSim models, we suggest using the `--editable` option with pip. 

## Tutorial

After installing CrossSim and its dependencies, get started by checking out the CrossSim [interactive tutorial](https://github.com/sandialabs/cross-sim/tree/main/tutorial) which shows off many of the new features in Version 3.0. The tutorial walks through how to use CrossSim's Numpy-like cores to drop it easily into application code. It also contains examples of how to model different physical effects, how to use different data mapping schemes, and how to define fully customizable physical device models.

## Neural network inference

CrossSim uses git submodules to distribute [neural network datasets and device lookup tables for training](https://github.com/sandialabs/cross-sim-data), and [neural network models](https://github.com/sandialabs/cross-sim-models). If you plan to use CrossSim's neural network inference scripts in the applications directory, after cloning this repository, the following commands will fetch the submodules.
```
git submodule init
git submodule update --progress
```

This command downloads about 1.2GB of data for the two repositories (combined), which is then de-compressed. After cloning the submodules, you can test CrossSim Inference by running the following commands:
```
cd applications/dnn/inference
python run_inference.py
```
This will run a CrossSim Inference simulation using a simple CNN and the MNIST dataset by default. The use of the run_inference.py script is explained in more detail in Chapter 3 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf). Example scripts to perform inference are in the ``inference`` directory and described in the same chapter of the manual. If you are interested in benchmarking large neural networks, a CUDA-capable GPU is recommended.

Due to file size and copyright, this distribution does not include the ImageNet dataset. Section 4.3 of the [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf) explains how to add your own copy of the ImageNet test set to CrossSim.


## Adding new device models
Users can add their own custom models for their resistive devices. Any device conductance errors that can be implemented as a Python function (including look-up tables) can be called as part of a CrossSim simulation. Custom device models can be added to the following directory, which contains several examples:
```
/simulator/devices/custom
```
See Part 3 of the [tutorial](https://github.com/sandialabs/cross-sim/tree/main/tutorial) as well as the ``simulator/devices`` page for a guide on how to do this.


## Adding new ADC models
Users can also add their own custom models for ADCs, by creating a compact model of the circuit that can be implemented as Python function (including look-up tables). Users can add their ADC models to the following directory, which contains several examples:
```
/simulator/circuits/adc
```
See the ``simulator/circuits/adc`` page for details on how to do this.

## Adding new neural network models, datasets, and device lookup tables
To use your own pretrained models and datasets for use with CrossSim Inference, or your own device experimental data for use with CrossSim Training, modify the url field in the .gitmodules folder. If you have no yet initialized the submodules, use the commands above. To change the submodule url after the submodules have been initialized, use the following commands:
```
git submodule sync
git submodule update --init --remote
```
Instructions for adding new neural network models and devices into CrossSim for use during inference can be found in Chapters 5 and 7 of [manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf) respectively.

If you would like to contribute your device data or models, create a pull request against this repository and the [data](https://github.com/sandialabs/cross-sim-data) or [pre-trained models](https://github.com/sandialabs/cross-sim-models) repositories.

## Citing CrossSim
If you use CrossSim for research please cite:
```text
@article{CrossSim,
  author={T. Patrick Xiao and Christopher H. Bennett and Ben Feinberg and Matthew J. Marinella and Sapan Agarwal},
  title={CrossSim: accuracy simulation of analog in-memory computing},
  url = {https://github.com/sandialabs/cross-sim},
}
```

## Developing CrossSim

When developing for CrossSim, different developer tools are used to improve code quality

### Linting
We use `ruff` to lint our code.
```
ruff check simulator/
```

### Type checking
We use `mypy`.
```
mypy simulator/
```

### Formatting
We use `black`.
```
black simulator/
```

### Documentation
We use `sphinx`.
```
sphinx-build -b html docs/sphinx/source/ docs/sphinx/build/
```
If a new module is added, it can be added to the docs using
```
sphinx-apidoc simulator/ -o docs/sphinx/source/
```

### Tests
We use `pytest` and `hypothesis`.
```
pytest
```

## Contact us
For questions, feature requests, bug reports, or suggestions, please submit a new issue through GitHub. The team will get back to you as soon as we can.
