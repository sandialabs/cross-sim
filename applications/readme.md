# Applications

This directory contains modules and scripts that are built on top of CrossSim's AnalogCore to run more complex applications.

The ``dnn`` directory contains modules to run neural network algorithms. For more details on neural network inference, see ``dnn/inference``. For neural network training, please use V2.0.

The ``dsp`` directory contains simple scripts that use CrossSim's AnalogCore to run Discrete Fourier Transforms, which are essential kernels for digital signal processing applications (though in this case, the processing is analog). The intention is for these scripts to be easy starting points for the user to adapt them into more complex application code.

The ``matlab`` directory shows a basic example of how to run CrossSim by calling Python from MATLAB.