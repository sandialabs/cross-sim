# Analog-to-digital converter (ADC)

This directory contains models for the ADC used to convert the analog MVM/VMM quantities to discrete digital values. These models are mainly selected and controlled by parameters inside the ``ADCParameters`` class (``params.xbar.adc.mvm`` or ``params.xbar.adc.vmm``, passed into the class as ``adc_params``).

The ``IADC`` class is the base class for ADC models. It natively implements the ``set_limits`` function that sets the range of the ADC, i.e. the analog inputs that correspond to the minimum and maximum digital levels. The method for determining the range is selected by ``adc_params.adc_range_option``. If using the ``MAX`` or ``GRANULAR`` option, the limits are computed from a simple rule based on the size of the matrix and/or the resolution of the cell conductances. These are documented in Section 9.3 of the [CrossSim Inference manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf). If using the ``CALIBRATED`` option, the ADC limits are set by the user as parameters, in ``adc_params.calibrated_range``. The ``set_limits`` function is usually only called once, upon setting the matrix to a core.

Specific ADC models inherit from ``IADC``, and define a ``convert`` function which quantizes the continuous-valued inputs during the simulation. The quantization should ideally round the inputs to equally spaced levels between the limits specified using ``set_limits``. The actual quantization behavior depends on the circuit implementation of the ADC and is itself subject to component errors in the ADC. The bit resolution of the ADC is set by ``adc_params.bits`` and determines the number of levels.

Specific ADC models can also extend the ``set_limits`` function to initialize variables that represent the physical state of the ADC.

An ADC model can be selected by setting ``params.xbar.adc.mvm.model`` or ``params.xbar.adc.vmm.model`` to an appropriate string that corresponds to the name of the ADC class.

### Generic ADC models

The generic ADC models implement the standard quantization behavior described above, assuming an ideal circuit implementation. There are two types:
- The ``QuantizerADC`` rounds the input to one of 2^_N_ levels between the min and max, where _N_ is the bit resolution. The min and max do not have to be symmetric about zero.
- The ``SignMagnitudeADC`` assumes that the digital outputs are in sign-magnitude representation. This means that the min and max must be symmetric about zero, and the number of levels is 2^(_N_-1) because there are two redundant representations of zero.

### Circuit-based ADC models

We provide four models of ADC circuits: the ramp ADC (``RampADC``), the successive approximation register (SAR) ADC (``SarADC``), the pipeline ADC (``PipelineADC``), and the cyclic ADC (``CyclicADC``). Each model inherits from ``IADC`` and is based on analytical equations that govern the behavior of the circuit. These circuits rely on switched capacitors, operational amplifiers, and comparators. Non-idealities can be modeled by introducing random mismatches in the capacitance values, random input offset voltages to the comparators, and finite gain to the amplifiers. The magnitude of these non-idealities can be set using parameters specific to the ADC type. For example,
```
params.xbar.adc.mvm.model = "SarADC"
params.xbar.adc.mvm.gain_db = 60
params.xbar.adc.mvm.sigma_capacitor = 0.15
params.xbar.adc.mvm.sigma_comparator = 0.10
params.xbar.adc.mvm.split_cdac = True
params.xbar.adc.mvm.group_size = 8
```
By selecting the ``SarADC`` model, the parameters of the ``SarADCParameters`` object will be set. The definitions of these parameters can be found in the ``SarADCParameters`` class which is defined in ``parameters/xbar_params.py``.

In general, these ADC models extend the ``set_limits`` function to randomly sample the errors and store them for the rest of the simulation. Then, the ``convert`` function implements the circuit's functionality based on analytical expressions. For more details, see Ref. [1].

### Adding a new custom ADC model

To add a new ADC model, follow these steps. See the circuit-based ADC model files mentioned above for examples of the appropriate imports and functional behavior.

- Create a new file that defines the new ADC class. The class can inherit from ``IADC`` or from one of the other ADC classes to inherit their behavior.
- Write a ``set_limits`` function that calls the ``set_limits`` of the parent class, then if needed, initializes the ADC state or errors based on parameters. This initialization need not be analytical as in the existing models, but can consult a look-up table to model more arbitrary behavior.
- Write a ``convert`` function that implements the quantization functionality of the ADC circuit. This also need not be analytical, but can consult a look-up table to model more arbitrary behavior.
- At the top of ``adc.py``, import the created ADC class from the new file.
- If needed, create a new parameters class that inherits from the ``ADCParameters`` object inside ``parameters/xbar_params``. Define and set the default values for the parameters.
- In ``PairedADCParameters``, add clauses to set the ``mvm`` or ``vmm`` attribute to the new ADC parameters object, following the existing examples.

You can now set the ADC model to the name of the created ADC class, by setting the appropriate ``model`` parameter as shown above.

#### References

[1] M. Spear, J. E. Kim, C. H. Bennett, S. Agarwal, M. J. Marinella, and T. P. Xiao, "The Impact of Analog-to-Digital Converter Architecture and Variability on Analog Neural Network Accuracy" _IEEE Journal on Exploratory Solid-State Computational Devices and Circuits_, 2023.