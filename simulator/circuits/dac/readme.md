# Digital-to-analog converter (DAC)

This directory contains models for the DAC used to quantize the MVM or VMM vector inputs to discrete values. These models are mainly selected and controlled by parameters inside the ``DACParameters`` class (``params.xbar.dac.mvm`` or ``params.xbar.dac.vmm``, passed into the class as ``dac_params``). The DAC models are simpler than the ADC models.

The ``IDAC`` class is the base class for DAC models. A DAC object can implement three functions:
- ``set_limits``, which sets the min and max output values to correspond to the ranges set in ``params.core.mapping.inputs.mvm.min``
- ``convert``, which quantizes the input vector values to equally spaced levels within the set range. The number of levels is based on the bit resolution in ``dac_params.bits``
- ``convert_sliced`` which implements input bit slicing. This function quantizes the input vector and returns a list of vectors corresponding to the vectors of input bit slices. The bit width of a slice is controlled by ``dac_params.slice_size``. The values in a slice should be one of 2^_B_ discrete values between 0 and 1, where _B_ is the slice size. The slice size should be less than the total bit resolution but the two do not have to be divisible. Physically, the slice size corresponds to the resolution of the DAC circuit. Different bit slices of the digital input are time-multiplexed into the DAC and an analog MVM is computed on one input bit slice at a time.

The generic DAC models implement the standard quantization behavior described above, assuming an ideal circuit implementation. There are two types:
- The ``QuantizerDAC`` rounds the input to one of 2^_N_ levels between the min and max, where _N_ is the bit resolution. The min and max do not have to be symmetric about zero.
- The ``SignMagnitudeDAC`` assumes that the digital signals are in sign-magnitude representation. This means that the min and max must be symmetric about zero, and the number of levels is 2^(_N_-1) because there are two redundant representations of zero. When implementing input bit slicing, only the magnitude bits are counted in the slice size. The sign bit is applied with each slice by allowing negative outputs, corresponding to negative input voltages to the array.

A DAC model for an MVM or VMM can be selected by setting a string corresponding to the name of the DAC object class.
```
params.xbar.dac.mvm.model = "SignMagnitudeDAC"
```

Custom DAC models can be added similarly to custom ADC models (see the README in the ``adc`` directory).