# CrossSim-PyTorch Integration

This directory contains a CrossSim wrapper for PyTorch layers to enable testing of neural networks using CrossSim's analog MVM simulation engine. Implemented layers are intended to be drop-in compatible with equivalent PyTorch layers and should behave similarly to the equivalent PyTorch layers when no CrossSim non-idealities are enabled. Note: though differences in the results will be small in this ideal case, **exactly identical numerical results between PyTorch layers and the CrossSim equivalents should not be expected.**

All implemented layers support adding the bias in either digital or analog by applying the bias across 1 or more rows within the array. Additionally, all implemented layers provide a gradient computation for CrossSim-in-the-loop training.

### Supported Layers:
- Linear
- Conv1d
- Conv2d
- Conv3d

In addition to implementation of PyTorch layers, this directory also provides utility functions for converting existing neural networks to use CrossSim equivalent layers and profiling the inputs and outputs of CrossSim layers for neural network calibration (e.g. to calibrate ADC ranges).

### Known Limitations:
- CrossSim analog layers do not support `torch.load` or `torch.save`. Use the `to_torch` function to convert analog layers to conventional torch layers before saving.
- Conv layers do not support dilated convolutions

## Using Analog PyTorch Layers
------
CrossSim analog layers can be directly instantiated like torch layers with the following convention:
`analog_layer = AnalogLayername(CrossSimParameters(), *base layer parameters, *analog specific parameters)`
For example:
`analog_layer = AnalogLinear(params, 2, 3, bias_rows=1)`
Would instantiate an AnalogLinear layer with in_features=2 and out_features=3 and the bias implemented within the array, using 1 row per core to store the bias.

CrossSim analog layers accept `device` and `dtype` arguments as in conventional PyTorch layers. If a layer is declared on a GPU, CrossSim must also use the GPU (i.e. `simulation.useGPU=True`). Datatypes (dtypes) will be mapped to equivalent cupy/numpy datatypes for the internal CrossSim layers according to the `_numpy_to_torch_dtype_dict` in [layer.py](layer.py).

All CrossSim analog layers maintain two copies of the weight and bias, that is, `layer.weight` will return an unmodified view of the weights in the layer and the AnalogCore object in each layer separately maintains the analog weights (see below) and bias (if using analog biases). Importantly, the ideal weights will not be used in inference operations, but they are used for gradient calculations and can be useful for debugging and layer error analysis. Additionally because the torch version of weights is maintained, most existing torch functions will still operate as expected on the layer; however, except for explicitly supported operations such as forward and backward computations these will not use the CrossSim view of the weights.

CrossSim will automatically keep the torch view of the weights and the core weights synchronized with two exceptions. First, using in-place operations on the torch view of the weights (e.g., `add_`) cannot be detected by CrossSim. Second, updating individual elements or groups of items though `__setitem__` will not automatically update the core weights. In either case `layer.synchronize()` should be called after the operation to bring the CrossSim version of weights back into consistency with the torch version. `synchronize` will resample initialization-time errors (e.g., programming error) in the analog array, so if a specific error pattern is desired it should be saved before calling `synchronize`.

The analog simulation parameters (CrossSimParamters) of an analog layer can be updated without rebuilding the layer using the `reinitialize()` function. This function is automatically called if either `layer.params` or `layer.bias_rows` is updated. `reinitialize()` directly rebuilds the underlying AnalogCore so it can also be used to resample the initialization-time errors. Importantly, even if no parameters impacting initialization-time errors are modified, any change to the internal params object will resample all initialization-time errors and should be handled as discussed for `synchronize` above.

CrossSim analog layers provide two methods for viewing the CrossSim version of weights. `get_matrix()` wraps the internal AnalogCore `get_matrix()` function and provides a view of the 2D matrix programmed into the array or arrays. `get_core_weights()` reshapes the 2D matrix into the same shape as the torch weight matrix. This is useful for analyzing element-wise error. `get_core_weights` is also used for analog-aware training to compute the gradients with respect to the programmed rather than ideal matrices.

## Working with Torch Models
------
The [convert.py](convert.py) files contains functions for working with full models rather than individual layers. When processing full models, these functions will only modify the CrossSim analog layers while the default torch layers will be unchanged. This includes full model wrappers for `reinitialize()` and `synchronize()` described above. These functions will call the layer-level function on each CrossSim layer.

### Converting Models to Analog
Existing neural network models can be converted to use CrossSim to support analog layers using the `from_torch` function. For example:
`analog_model = from_torch(model, params)`
Will return a copy of the model with all supported layers replaced by analog equivalents using the params object specified by params. The process can be reversed using:
`reconstructed_model = to_torch(analog_model)`

For models where a different CrossSimParameters object is expected per layer (e.g., for ADC range calibration) a list of CrossSimParameters objects can be used instead. For models which contain other models rather than just layers, CrossSim performs a recursive traversal of the models during conversion. This traversal order matches the order of `convertible_modules()` which should be used to ensure that per-layer parameters are correctly matched with existing layers. A list arguments can also be used for bias_rows in `from_torch()` and physical_weights in `to_torch()`.

### Simulating Inference with CrossSim
After converting an existing PyTorch model to use CrossSim, the resulting model can be used similarly to the original PyTorch model in order to simulate end-to-end inference, where the supported layers are processed via simulated analog MVMs inside the AnalogCores. For example, the simulation can be run using: `output = analog_model(input)`
where `input` is the input data, and `output` is the simulated output of the analog inference accelerator. Consistent with standard PyTorch model usage, `input` should reside on the same device as the model, and the first dimension of `input` is the batch.

### Analog-Aware Training with CrossSim
Models containing analog layers can be directly integrated into existing training flows with a minor change. After the optimizer update step, call `synchronize(model)` since the built-in torch optimizers use in-place updates. An example of this can be see in section 2.3 of [this tutorial](../../../tutorial/NICE24/tutorial_pt2.ipynb).

## Adding New Analog PyTorch Layers
------
New analog PyTorch layers should inherit from both the baseline torch layer and the CrossSim torch AnalogLayer in [layer.py](layer.py) in that order. The full list of attributes that should be included in a new analog torch layer can be found in [layer.py](layer.py). Implementing classes should also implement the following functions: `form_matrix`, `reinitialize`, `from_torch`, `to_torch` and `forward` as defined in [layer.py](layer.py). 

Layer functionality that is not specific to PyTorch should be implemented based on the AnalogLayer base class in [analog_layer.py](../analog_layer.py). Torch implementations should focus on torch-specific behavior and/or interfacing with torch, for instance converting between numpy arrays and torch tensors. For compatibility with both CPU and GPU implementations, wrap all numpy arrays coming from CrossSim in `torch.from_dlpack()` and ensure that all Torch tensors are detached before passing them into CrossSim.

To support gradient computation, the forward function should call into a separate AnalogGradient class which inherits from `torch.autograd.Function`. This class should implement both the forward call (again deferring all non-torch specific behavior to the core object) and a backward function. To improve the convergence of analog-aware training the forward function should save the CrossSim view of the weights (from `get_core_weights`) which will be used for the gradient calculation in the backward pass.

The torch [AnalogLinear](linear.py) class is a good example of how to implement new layers.
