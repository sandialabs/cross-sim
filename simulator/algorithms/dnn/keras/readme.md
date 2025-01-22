# CrossSim-Keras Integration

This directory contains a CrossSim wrapper for Keras layers to enable testing of neural networks using CrossSim's analog MVM simulation engine. Implemented layers are intended to be drop-in compatible with equivalent Keras layers and should behave similarly to the equivalent Keras layers when no CrossSim non-idealities are enabled. **Note: identical numerical results between Keras layers and the CrossSim equivalents should not be expected.**

All implemented layers support adding the bias in either digital or analog by applying the bias across 1 or more rows within the array.

### Supported Layers:
- Dense
- Conv1D
- Conv2D
- Conv3D
- DepthwiseConv1D
- DepthwiseConv2D

In addition to implementation of keras layers, this directory also provides utility functions for converting existing neural networks to use CrossSim and profiling the inputs and outputs of CrossSim layers for neural network calibration.

### Known Limitations:
- CrossSim analog layers do not support `keras.saving.load_model`, `keras.save.save_model`, or `model.save`. Use the `to_keras` function to convert analog layers to conventional keras layers before saving.
- Conv layers do not support dilated convolutions
- Analog layers do not support gradient computations
- Functional models that use `operations` rather than `layers` cannot be converted using the `convert` function

## Using Analog Keras Layers
------
CrossSim analog layers can be directly instantiated like keras layers with the following convention:
`analog_layer = AnalogLayerName(**cross sim specific argument dict, **base layer argument dict)`
For example:
`analog_layer = AnalogDense(params=CrossSimParameters(), units = 2)`
Would instatitate an AnalogDense layer with 2 units (outputs) using a default CrossSimParameters() object. As with standard keras layers the layer is not build until an input spec is specified or the first operation is run. Importantly, all arguments must be referred to by keyword rather than positionally.

All CrossSim analog layers maintain two copies of the weight and bias, that is, `layer.get_weights()` will return an unmodified view of the weights (including bias) in the layer and the AnalogCore object in each layer separately maintains the analog weights (see below) and bias (if using analog biases). Importantly, the ideal weights will not be used in inference operations, but they can be useful for debugging and layer error analysis. Additionally because the keras version of weights is maintained, most existing keras functions will still operate as expected on the layer; however, except for explicitly supported operations such as forward and computations these will not use the CrossSim view of the weights.

CrossSim will automatically keep the keras view of the weights and the core weights synchronized when `layer.set_weights()` is called. Other methods of changing weights will not be visible from CrossSim and require special handling to ensure consistency.

The analog simulation parameters (CrossSimParamters) of an analog layer can be updated without rebuilding the layer using the `reinitialize()` function. `reinitialize()` directly rebuilds the underlying AnalogCore so it can also be used to resample the initialization-time errors. Importantly, even if no parameters impacting initialization-time errors are modified, any change to the internal params object will resample all initialization-time errors.

CrossSim analog layers provide two methods for viewing the CrossSim version of weights. `get_matrix()` wraps the internal AnalogCore `get_matrix()` function and provides a view of the 2D matrix programmed into the array or arrays. `get_core_weights()` reshapes the 2D matrix into the same shape as the keras weight matrix. This is useful for analyzing element-wise error.

## Working with Keras Models
------
The [convert.py](convert.py) files contains functions for working with full models rather than individual layers. When processing full models, these functions will only modify the CrossSim analog layers while the default keras layers will be unchanged. This includes a full model wrapper for `reinitialize()` described above. These functions will call the layer-level function on each CrossSim layer.

### Converting Models to Analog
Existing neural network models can be converted to use CrossSim to support analog layers using the `from_keras` function. For example:
`analog_model = from_keras(model, params)`
Will return a copy of the model with all supported layers replaced by analog equivalents using the params object specified by params. The process can be reversed using:
`reconstructed_model = to_keras(analog_model)`

For models where a different CrossSimParameters object is expected per layer (e.g., for ADC range calibration) a list of CrossSimParameters objects can be used instead. For models which contain other models rather than just layers, CrossSim performs a recursive traversal of the models during conversion. This traversal order matches the order of `convertible_layers()` which should be used to ensure that per-layer parameters are correctly matched with existing layers. A list arguments can also be used for bias_rows in `from_keras()` and physical_weights in `to_keras()`.

### Simulating Inference with CrossSim
After converting an existing KEras model to use CrossSim, the resulting model can be used similarly to the original keras model in order to simulate end-to-end inference, where the supported layers are processed via simulated analog MVMs inside the AnalogCores. For example, the simulation can be run using: `output = analog_model.predict(input)` 
where `input` is the input data, and `output` is the simulated output of the analog inference accelerator. CrossSim layers can support either `channels_first` or `channels_last` data formats for convolutional layers.


## Adding New Analog Keras Layers
------
New analog Keras layers should inherit from both the  CrossSim keras AnalogLayer in [layer.py](layer.py) and the base keras layer in that order . Implementing classes should also implement the following functions: `call`, `build`, `form_matrix`, `get_core_weights`, and `reinitialize` as defined in [layer.py](layer.py). 

Layer functionality that is not specific to keras should be implemented based on the AnalogLayer base class in [analog_layer.py](../analog_layer.py). Keras implementations should focus on keras-specific behavior and/or interfacing with keras, for instance reordering the dimensions. Keras layers using CrossSim on the CPU (`params.simulation.useGPU = False`) can directly pass numpy arrays to and from CrossSim without issues; however if CrossSim is running on the GPU (`params.simulation.useGPU = True`) cupy arrays returned by CrossSim need to be passed using dlpack and gpu streams must be synchronized. For instance the following code from [conv.py](conv.py) shows how the return value out from a CrossSim core must be processed when using a GPU:
```python
out = self.core.apply(input_)
if self.useGPU:
    out = from_dlpack(out.toDlpack())
    self.stream.synchronize()
```

The keras [AnalogDense](dense.py) class is a good example of how to implement new layers.