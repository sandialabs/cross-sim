# CrossSim (V3.1)

CrossSim is a GPU-accelerated, Python-based crossbar simulator designed to model analog in-memory computing for any application that relies on matrix operations: neural networks, signal processing, solving linear systems, and many more. It is an accuracy simulator and co-design tool that was developed to address how analog hardware effects in resistive crossbars impact the quality of the algorithm solution.

CrossSim has a Numpy-like API that allows different algorithms to be built on resitive memory array building blocks. CrossSim cores can be used as drop-in replacements for Numpy in application code to emulate deployment on analog hardware. It also has a special interface to model analog accelerators for neural network inference.

CrossSim can model device and circuit non-idealities such as arbitrary programming errors, conductance drift, cycle-to-cycle read noise, and precision loss in analog-to-digital conversion (ADC). It also uses a fast, internal circuit simulator to model the effect of parasitic metal resistances on accuracy. For neural network inference, it can simulate accelerators with significant parameterizability at the system architecture level and can be used to explore how design choices such as weight bit slicing, negative number representation scheme, ADC ranges, and array size affect the sensitivity to these analog errors. CrossSim can be accelerated on CUDA GPUs, and inference simulations have been run on large-scale, industry-standard deep neural networks such as ResNet50 on ImageNet. CrossSim's simulation speed on ResNet50 is within ~3X of TensorFlow-Keras using baseline (simplest) analog hardware simulation settings.

 __To simulate neural network training, please use [CrossSim Version 2.0](https://github.com/sandialabs/cross-sim/releases/tag/v2.0). Support for neural network training will be brought back in a future software update.__

CrossSim does not explicitly model the energy, area, or speed of analog accelerators. 

## Requirements
CrossSim has been tested on Ubuntu 18.04 and Windows 10 using Python 3.11.6.

CrossSim requires the following Python packages:
* Numpy 1.26.3
* SciPy 1.11.4
* TensorFlow 2.17.0 (for DNN inference)
* PyTorch 2.2.1 (for DNN inference/hardware-aware training)
* IPython 8.8.0 (for tutorials)
* MatPlotLib 3.8.2 (for tutorials and application examples)
* Cupy 12.3.0 with CUDA 12.3, or CuPy 8.3.0 with CUDA 10.2 (for GPU acceleration, intermediate CuPy versions will likely work)

Several of the neural network models provided with CrossSim may require additional packages.
ImageNet models may require:
* OpenCV 4.5.4
* Torchvision 0.11.1

CrossSim has been tested with the version numbers above, but other versions may work as well.

## Installation

CrossSim can be installed using pip using the following command from the base repository directory.
```
pip install .
```
If you plan to modify the internal internal CrossSim models, we suggest using the `--editable` option with pip. 

## Tutorial

After installing CrossSim and its dependencies, get started by checking out the CrossSim [interactive tutorial](https://github.com/sandialabs/cross-sim/tree/main/tutorial) which shows off many of the features in Version 3.0. The tutorial walks through how to use CrossSim's Numpy-like cores to drop it easily into application code. It also contains examples of how to model different physical effects, how to use different data mapping schemes, and how to define fully customizable physical device models. Some newer tutorials presented at [ISCA 2024](https://github.com/sandialabs/cross-sim/tree/main/tutorial/ISCA2024) and [NICE 2024](https://github.com/sandialabs/cross-sim/tree/main/tutorial/NICE2024) show off some of the new features of Version 3.1, including the PyTorch interface. 

## Neural network interface

As of Version 3.1, CrossSim has a new interface that integrates with both [PyTorch](https://github.com/sandialabs/cross-sim/tree/main/simulator/algorithms/dnn/torch) and [Keras](https://github.com/sandialabs/cross-sim/tree/main/simulator/algorithms/dnn/keras). Both interfaces can take a neural network model and replace the layers that are compatible with analog in-memory processing (convolutions and fully-connected layers) with new analog layer types that behave similarly to their PyTorch/Keras equivalents, but which process matrix operations using CrossSim's AnalogCores. This can be used to easily simulate deep neural network (DNN) inference on pre-trained PyTorch/Keras models. The PyTorch interface additionally supports backpropagation, enabling hardware-aware DNN training using user-specified analog hardware parameters. Examples of usage are included in application-level example scripts (for [PyTorch](https://github.com/sandialabs/cross-sim/tree/main/applications/dnn/torch) and [Keras](https://github.com/sandialabs/cross-sim/tree/main/applications/dnn/keras)), the [ISCA tutorial](https://github.com/sandialabs/cross-sim/tree/main/tutorial/ISCA2024), and the [NICE tutorial](https://github.com/sandialabs/cross-sim/tree/main/tutorial/NICE2024).

## Provided datasets and pre-trained models

Users can optionally download some provided example data using CrossSim's associated git submodules: [neural network datasets and device lookup tables for training](https://github.com/sandialabs/cross-sim-data), and some example pre-trained [neural network models](https://github.com/sandialabs/cross-sim-models). After cloning this repository, the following commands will fetch the submodules.
```
git submodule init
git submodule update --progress
```

This command downloads about 1.2GB of data for the two repositories (combined), which is then de-compressed.

## Adding new device models
Users can add their own custom models for their resistive memory devices. Any device conductance errors that can be implemented as a Python function (including look-up tables) can be called as part of a CrossSim simulation. Custom device models can be added to the following directory, which contains several examples:
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


## Citing CrossSim
If you use CrossSim for research please cite:
```text
@article{CrossSim,
  author={Ben Feinberg and T. Patrick Xiao and Curtis J. Brinker and Christopher H. Bennett and Matthew J. Marinella and Sapan Agarwal},
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
