# Algorithms

This directory contains code that implement higher-level algorithmic constructs that build on the analog MVM/VMM functionality of ``AnalogCore``. These files implement algorithmic building blocks that can be conveniently used by a variety of application algorithms. Application code can then be simplified by calling these building blocks rather than directly using ``AnalogCore``.

- The ``dnn`` module contains the `DNN` class, which is CrossSim's internal representation of a deep neural network, which is essentialy a graph that connects different layers and operations together. The `Convolution` object implements the convolution layer of a neural network. Fully-connected layers are implemented using ``AnalogCore`` directly.

- The ``dsp`` module currently contains a single `DFT` class that implements an analog Discrete Fourier Transform. The DFT can be easily implemented using MVM as shown in the tutorial, so this mainly serves as a simple example of building an algorithmic building block on top of ``AnalogCore``.