# Configs

This directory contains simulation configurations for several use cases of interest. These configurations are ``CrossSimParameters`` objects that have been exported to JSON files. To load a configuration, do the following:
```
params = CrossSimParameters()
params = params.from_json("config_name.json")
```
The ``params`` can then be used to set an ``AnalogCore`` or to replace the call to ``set_params()`` for neural network inference.

The ``default.json`` file contains the default CrossSim parameters. The other JSON files contain parameters that correspond to existing analog accelerator architectures in the literature.

### ISAAC

`ISAAC.json` implements the ISAAC accelerator [1]. Notable settings include:
- 16-bit weights with 2 bits/cell slicing
- 16-bit inputs with 1-bit slices
- 128 x 128 crossbar array
- Offset mapping, with analog subtraction of the offset using a unit column
- Resistive RAM devices between 2 kOhm and 2 MOhm resistance. Devices are assumed to be error-free in this configuration.
- 9-bit ADC with ``GRANULAR`` range option*
- Separate digitization of every input bit slice

A difference between this model and the original paper is that ISAAC uses a "flipped" weight mapping scheme to reduce the ADC resolution to 8 bits while still using the ``GRANULAR`` setting. This is not  currently implemented by CrossSim. ISAAC uses a SAR ADC. This configuration assumes the SAR ADC is free of component errors, so the the generic ADC models are used. If the user wishes to simulate SAR ADC component errors, please use the ``ISAAC-SarADC.json`` file and set the desired component errors. Note that for large simulations with hundreds to thousands of arrays, setting up the random errors in the individual SAR ADCs may be time consuming.

### PRIME

``PRIME.json`` implements the PRIME accelerator [2]. Notable settings include:
- 8-bit weights with 4 bits/cell slicing
- 6-bit inputs with 3-bit slices (i.e. 3-bit DAC)
- 256 x 256 crossbar array
- Differential (balanced) mapping
- Resistive RAM devices between 1 kOhm and 20 kOhm resistance. Devices are assumed to be error-free in this configuration.
- 6-bit ramp ADC with ``CALIBRATED`` range option*
- Separate digitization of positive and negative crossbar
- Separate digitization of every input bit slice

PRIME requires ADC ranges to be calibrated to a specific application before high accuracy can be obtained. See ``applications/dnn/inference`` for how to find these calibrated ranges. To simulate PRIME without the ADC, set ``adc_bits = 0``.

### SONOS-TCAS

``SONOS-TCAS.json`` implements the SONOS-based accelerator in [3]. Notable settings include:
- 8-bit weights, no bit slicing
- 8-bit inputs with 1-bit slices
- 1152 x 256 crossbar array
- Differential (balanced) mapping
- SONOS devices between 35 kOhm and 6 TOhm resistance. Uses the ``SONOS`` programming error and drift models.
- 8-bit ramp ADC with ``CALIBRATED`` range option*
- Analog subtraction of positive and negative column currents
- One digitization for all input bit slices (analog shift-and-add accumulation)

As with PRIME, the SONOS accelerator requires ADC ranges to be calibrated to a specific application before high accuracy can be obtained.

#### References

[1] A. Shafiee, A. Nag, N. Muralimanohar, R. Balasubramonian, J. P. Strachan, M. Hu, R. S. Williams, and V. Srikumar, "ISAAC: A convolutional neural network accelerator with in-situ analog arithmetic in crossbars," _International Symposium on Computer Architecture (ISCA)_, pp. 14–26, 2016.

[2] P. Chi, S. Li, C. Xu, T. Zhang, J. Zhao, Y. Liu, Y. Wang, and Y. Xie, “PRIME: A novel processing-in-memory architecture for neural network computation in ReRAM-based main memory,” _International Symposium on Computer Architecture (ISCA)_, pp. 27-39, 2016.

[3] T. P. Xiao, B. Feinberg, C. H. Bennett, V. Agrawal, P. Saxena, V. Prabhakar, K. Ramkumar, H. Medu, V. Raghavan, R. Chettuvetty, S. Agarwal, and M. J. Marinella, "An Accurate, Error-Tolerant, and Energy-Efficient Neural Network Inference Engine Based on SONOS Analog Memory," _IEEE Transactions on Circuits and Systems I: Regular Papers_, vol. 69, no. 4, pp. 1480-1493, 2022.