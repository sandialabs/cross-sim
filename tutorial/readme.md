# CrossSim V3.0 Tutorial

This tutorial demonstrates how to use CrossSim, particularly the new features that are introduced in the Version 3.0 release. The tutorial does not introduce all of CrossSim's features or modules but should be useful for getting started and for understanding the many use cases where our tool can be applied.

The tutorial is split up into three IPython Notebooks. We recommend going through these in order.
- Part 1 shows how to use CrossSim's AnalogCore, which has an easy-to-use Numpy-like API. It contains two examples of using the AnalogCore for signal processing and linear algebra applications. This part assumes fully ideal hardware settings to demonstrate equivalence to Numpy.
- Part 2 shows how to use CrossSim to investigate the effects of device and circuit non-idealities on the accuracy of the algorithm, and how these interact with architectural design choices such as how matrix values are mapped to arrays.
- Part 3 shows how to use CrossSim's interface to add arbitrary user-specified models for the behavior of the resistive memory device.

The tutorial notably does not show how to use CrossSim's neural network inference interface. This interface does not require the user to directly use CrossSim's AnalogCore, and is trickier to demonstrate in a Notebook. To learn more about simulating inference, go to the ``applications/dnn/inference`` directory or see the [CrossSim Inference manual](https://github.com/sandialabs/cross-sim/blob/main/docs/CrossSim_Inference_manual_v2.0.pdf) (from V2.0).