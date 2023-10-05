# Neural network inference

This directory contains CrossSim's neural network inference interface. For a full documentation of CrossSim Inference, please see the [inference manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf). The manual is currently being updated, but is still mostly up-to-date.

CrossSim Inference imports pre-trained neural networks from TensorFlow-Keras model files and builds its own internal graph of the neural network. Fully-connected and convolution layers are mapped onto a collection of CrossSim AnalogCores for analog hardware emulation. The intention is to provide a simple interface to the user that fully handles the construction of a deep neural network from elementary MVM operations. 

The inference module supports a diverse set of MLPs and CNNs, including non-sequential networks like ResNets, InceptionNets, and MobileNets. CrossSim comes with a number of datasets and pre-trained neural networks already available (e.g. for MNIST, CIFAR-10, CIFAR-100, and ImageNet). Users can add new custom datasets and Keras models: see Chapters 4 and 5 of the manual.

Notable changes to CrossSim Inference in V3.0:
- There is a large performance improvement for convolutional neural network (CNNs) simulations, when not using read noise or parasitic resistance models. For the simplest analog hardware simulation settings on an Nvidia A100 GPU, CrossSim simulates ImageNet/ResNet-50 at a speed of 23.0 ms/image.
- There is an option to inspect the analog accelerator configuration including the number and size of arrays used in all neural network layers. To enable, set ``show_HW_config = True`` in ``inference_config.py``.
- In addition to classification accuracy, the outputs from the last layer of the neural network can be returned and saved. This is valuable for a variety of use cases, such as regression problems. To enable, set ``return_network_outputs = True`` in ``inference_config.py``.
- ADC ranges and profiled ADC inputs are all in units of _normalized current_, rather than algorithm units, so that they are more useful to hardware designers. A normalized current of ``1`` means the MVM summed current is equal to 1x the maximum current through a single device (i.e. max conductance and max voltage).

## Running inference simulations

Neural network inference is run by specifying a configuration file, then running an inference execution script.

First, install the submodules containing benchmark datasets and pre-trained models. For instructions, see the [top-level README](https://github.com/sandialabs/cross-sim).

The ``inference_config.py`` file is the configuration file containing the settings for the inference simulation, including the choice of dataset, neural network, and the configuration of the simulated analog hardware. Copies of this file can be made to be used as independent configuration files for different inference simulations.

The ``run_inference.py`` file is the script that executes the inference simulation. Select the desired configuration file by using the ``import`` statement at the top of this file. Then, run the following command inside this directory to run an inference simulation:

```
python run_inference.py
```

The ``run_inference_errorloopy.py`` file shows an example of how to modify the simulation script to perform a parameter sweep. This can be used to investigate how the accuracy of an analog inference accelerator varies with one or more device, circuit, architecture, or algorithm design variables.

## ADC limits and profiling

One of the trickiest parts of the analog accelerator to simulate is the analog-to-digital converter (ADC). When simulating a new neural network, it is generally not obvious how to set the resolution and range of the ADCs, which can vary from layer to layer. We provide several options.

To disable the ADC, set ``adc_bits`` to zero.

To enable the ADC without having found the optimal ADC ranges, the user can set the ``adc_range_option`` option to ``MAX`` or ``GRANULAR``. These are simple heuristic methods that are well-known in the literature to compute the ADC range for every layer by using only the shape of the weight matrix. However, being simple, they can lead to significant accuracy loss if the ADC resolution is not high enough. See Chapter 9.3 of the inference manual for details.

To achieve high accuracy at the lowest possible ADC resolution, the range of each layer's ADC must be separately optimized [1]. This is a complicated process, but CrossSim offers a streamlined way to do this:
- First, run simulations with the ADC disabled to profile the ADC inputs, and save this data to disk. Use the ``run_inference_profiling.py`` file to help with this. The profiled inputs are saved in the ``adc`` folder.
- Use the statistics of the ADC inputs collected from these simulations to determine the ADC range for every layer that minimizes both quantization error and clipping error. Some example scripts for performing this function are provided in the ``adc`` folder. The user can also write their own. These scripts save the optimized ADC limits to named files in the ``adc/adc_limits`` folder.
- The optimized ADC ranges can then be loaded when running a simulation with the corresponding hardware settings and with ``adc_range_option`` set to ``CALIBRATED.`` The limits file can be added inside ``interface/dnn_setup.py``.

More details on this process can be found in Section 9.4 of the manual.

The optimized ADC limits are only valid for a specific combination of neural network and hardware settings. The ``adc_limits`` directory contains calibrated ADC limits for a number of neural networks that are part of the ``cross-sim-models`` submodule, but does not account for every possible simulation configuration with these models. The most extensive set of ADC limits we have generated are for the ResNet50-v1.5 network (part of the MLPerf Inference benchmark). These can be found inside the ``adc_limits/imagenet`` and ``adc_limits/imagenet_bitslicing`` folders. See ``interface/dnn_setup.py`` to see the simulation settings that are matched to these different calibrated ADC limits.

[1] T. P. Xiao, B. Feinberg, C. H. Bennett, V. Prabhakar, P. Saxena, V. Agrawal, S. Agarwal, and M. J. Marinella, "On the accuracy of analog neural network inference accelerators," _IEEE Circuits and Systems Magazine_, 22(4), pp. 26-48, 2022.