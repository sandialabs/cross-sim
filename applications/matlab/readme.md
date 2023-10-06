# CrossSim from Matlab

This directory contains an example of how to use CrossSim's analog models from Matlab. This uses of Matlab's native support for calling [Python libraries](https://www.mathworks.com/help/matlab/call-python-libraries.html).

To use this interface CrossSim should be installed on the system PYTHONPATH. See the instructions in the [top-level README](https://github.com/sandialabs/cross-sim) for installation instructions.

An example of the full functionality of the interface is shown in `matlab_example.m`. CrossSim in Matlab uses a subset of the `AnalogCore` interface. An `AnalogCore` is created from a matrix and a `CrossSimParameters` object which can either by loaded from json and/or directly modified.

Once an `AnalogCore` has been created it can be multiplied using the `xbar.dot` function with either 1D or 2D matrices, and the transpose can be accessed using `xbar.T`. Similar using the `AnalogCore` as the right-hand side value uses the `xbar.rdot` function.

Importantly, this does not support multiplication and updating based on portions of the matrix using Matlab's slicing syntax. To update the full matrix simulated by the analog core use `xbar.set_matrix`. This matrix must match the size of the original matrix.

## Supporting Matlab's '\*' Operator
By default CrossSim does not support the native Matlab '\*' for matrix multiplication. Since Python uses the '\*' operator for element-wise multiplication overloading it for Matlab compatibility creates a potential source of error in python. However, if CrossSim installation will be used only with Matlab, AnalogCore can be modified with the provided patch to overload the '\*' operator. If the installation will be used for both Python and Matlab and the Matlab operator is desired, consider installing an unmodified version of CrossSim within a virtual environment.

To apply the provided patch run the following command from this directory. This must be run before CrossSim is installed or with CrossSim installed using `--editable`.
```
patch ../../simulator/cores/analog_core.py operator.patch
```

The patch can be reverted with the following command:
```
patch -R ../../simulator/cores/analog_core.py operator.patch
```

With this installed the following code can be used:
```matlab
params = py.simulator.CrossSimParameters();
matrix = rand(4);
vector = rand(4,1);
xbar = AnalogCore(matrix, params);

% Matlab:
matrix * vector
% CrossSim:
xbar * vector

% Matlab:
matrix' * vector
% CrossSim:
xbar.T * vector

% Matlab:
vector' * matrix
% CrossSim:
vector * matrix

% Matlab:
matrix * matrix
% CrossSim:
xbar * matrix

% Matlab:
matrix * matrix
% CrossSim:
matrix * xbar
```