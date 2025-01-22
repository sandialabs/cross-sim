# CrossSim Neural Network Integrations

CrossSim provides implementations of neural network layers which wrap the AnalogCore simulator interface. This directory contains the base implementations of these layers for use by the PyTorch and Keras neural network frameworks. The implementations in this directory should generally not be directly used but rather instantiated by PyTorch or Keras layer implementations. These implementations assume that each layer is implemented with a single AnalogCore; however, the layer can be partitioned across multiple arrays within the AnalogCore. Additionally, some layer operations (notably convolutions) require multiple MVM operations with internal buffering and indexing.

**Note: This directory also contains the CrossSim 3.0 neural network interface. This interface is deprecated and will be removed in the CrossSim 3.2 Release. The new PyTorch and Keras interfaces provide a superset of this functionality and users of the 3.0 interface are encouraged to port their applications to one of the new interfaces.**

### Supported Layers:
- Linear
- 1-3D Conv Layers (including depthwise and grouped convolutions)

### Known Limitations:
- Conv layers do not support dilated convolutions

## Adding New Layers
------
- New layers should subclass `AnalogLayer` and at a minimum implement the `form_matrix` and `apply` methods for a layer as well as a layer-specific `__init__`. 
- The `__init__` and `apply` functions should also provide profiling hooks to capture layer inputs and initialize storage for profiling adc inputs. 
- AnalogLayer provides basic implementations of other functions which are themselves thin wrappers around functions and attributes of the internal AnalogCore.

By convention, CrossSim layer implementations broadly follow PyTorch for dimension ordering and parameters naming.
