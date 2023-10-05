%
% Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
% Sandia Corporation, the U.S. Government retains certain rights in this software.
%
% See LICENSE for full license details
%

% Basic example of how to use the CrossSim matrix interface directly from Matlab
% This uses the CrossSim AnalogCore interface which provides a simple matrix
% multiplication interface that internally models an analog array

% AnalogCores use a CrossSimParameters object to configure the internal state
% These can either be created directly and modified or loaded from a JSON file

% Creates a default parameters object
offset_params = py.simulator.CrossSimParameters();
% Parameters can then be modified by directly modifying the python parameters object
% For example this line changes the core type from a balanced to offset representation
% for neative numbers/
offset_params.core.style = "OFFSET";
% For more information on parameters, please see the python parameters documentation

% Alternatively you can load a configuration from a JSON file
balanced_params = py.simulator.CrossSimParameters.from_json("example.json");
% The resulting object can also be modified
balanced_params.core.style = "BALANCED";

% Now we need a matrix to program into our array and a vector for multiplication
matrix = rand(4)
vector = rand(4,1);

% The AnalogCore object will behave exactly like a Matlab matrix but
% simulating analog non-idealities
xbar = py.simulator.AnalogCore(matrix, balanced_params);

% We can print an xbar to see the internal matrix representation
xbar.get_matrix()
% Just "xbar" works as well, but it will print out additional Python state information
xbar

% And we use the array for matrix vector multiplication
% Matlab:
matrix * vector
%CrossSim:
xbar.dot(vector)

% Or matrix matrix multiplication
% Matlab:
matrix * matrix
% CrossSim:
xbar.dot(matrix)

% We can also do a transpose for either operation
% Matlab:
matrix' * vector
% CrossSim:
xbar.T.dot(vector)

% Matlab:
matrix' * matrix
% CrossSim:
xbar.T.dot(matrix)

% Alternatively if you want to use the multiplied value on the left hand size use
% the rdot() function. Note that python handles 1D arrays slightly differently so
% we don't need to transpose them 
% Matlab:
vector' * matrix
% CrossSim:
xbar.rdot(vector)

% Matlab:
matrix' * matrix
% CrossSim:
xbar.rdot(matrix')

% We can update the matrix in our AnalogCore. Note however, the new matrix
% must be the same size as the previous one
new_matrix = rand(4);
xbar.set_matrix(new_matrix);

new_matrix
xbar.get_matrix()

% Finally, if you want to use the result of a CrossSim computation in a later Matlab
% function, you can use Matlab's double() function to convert it into a Matlab matrix
% Matlab:
new_matrix * vector
% CrossSim:
double(xbar.dot(vector))

% And thats it, with these basic functions you can use CrossSim's analog matrix
% multiplication simulator from Matlab.
% Two final notes:
% 1) This interface only supports a subset of CrossSim's AnalogCore functionality
% Specifically, operations like array slicing and indexing for operations and updates
% are not supported.
% 2) The above examples all use the dot() and rdot() functions for operations
% The README in this directory provides discusses how the native Matlab "*" operator
% can be used instead. This is not recommended for installations where both the Matlab
% and native Python interfaces will be used because this can introduce unexpected
% behavior in Python, but may be useful if primarily Matlab will be used.
% See the README for more details on configuration and potential pitfalls.