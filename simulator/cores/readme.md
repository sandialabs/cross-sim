# Cores

CrossSim's cores are the objects that represent analog in-memory computing arrays. Like standard memory arrays, each core has a method to program an array of values (``set_matrix``) and read out the values (``read_matrix``). These cores can also simulate MVMs and VMMs that are physically processed inside the memory arrays.

There is a three-level hierarchy of cores based on the level of abstraction. Lower-level cores are attributes of the higher-level cores. The highest level core (``AnalogCore``) is designed to be treated like a Numpy array, can handle negative and even complex-valued operands, and can be easily integrated with application software. The lowest level core (``NumericCore``) embodies a single physical resistive crossbar and is implemented using models of physical hardware.

### AnalogCore

The AnalogCore is the highest level core. Its ``set_matrix`` method can take as input a numerical matrix of arbitrary size and the most flexible numerical ranges. The following are some of AnalogCore's main features:
- It implements a Numpy-like API and behaves like a Numpy array, so that it can be easily integrated with application software. Please see Tutorial part 1 for a demonstration of how to create AnalogCores and use its API.
- It spatially partitions a large matrix into constituent cores, based on the maximum size along the row and column dimensions of the crossbar array. These constituent cores are of the ``BalancedCore``, ``OffsetCore``, or ``BitslicedCore`` type and can be accessed using ``AnalogCore.cores``.
- It decomposes complex-valued matrices into arrays of purely real numbers and handles MVMs between complex-valued matrices and vectors.

### BalancedCore, OffsetCore, and BitSlicedCore

These cores handle two aspects of how a matrix is mapped to a resistive crossbar: negative number handling and precision handling. Each of these cores takes a real-valued matrix as input to its ``set_matrix`` method, with dimensions equal to or smaller than the physical crossbar array.

- ``BalancedCore`` represents positive and negative matrix values by encoding each value in the difference in conductance of two resistive memory devices. During an MVM/VMM, the dot products with the negative devices are subtracted from the dot products with the positive devices. In ``BalancedCore``, the bit precision of the weight is not sliced, i.e. the total number of possible numerical levels is the linear sum of the number of accessible conductance levels in the two devices. 
	- Several parameters (in ``params.core.balanced``) control the behavior of ``BalancedCores``, such as whether subtraction occurs before or after digitization, and whether currents flowing through positive and negative devices can cancel along a column or row.
- ``OffsetCore`` represents a positive or negative matrix value using a single positive conductance, by encoding the value with an added offset. The midpoint of the conductance range represents zero, while conductances above and below the midpoint encode positive and negative values, respectively. During an MVM/VMM, the offset is computed by taking a scalar product of the vector with the offset, and this is subtracted from the MVM results. Like ``BalancedCore``, the precision is not sliced, i.e. the total number of numerical levels is equal to the number of accessible conductance level in a single device.
	- The parameter ``params.core.offset.style`` controls whether the offset is computed digitally, or in the analog domain using an extra column (MVM) or row (VMM).
- ``BitSlicedCore`` divides the bits of precision of the matrix values into slices of different significance. This can allow an arbitrarily high-precision matrix to be mapped onto several arrays of devices with relatively few programmable conductance levels each. Within each slice, positive and negative values can be represented using differential cells (like ``BalancedCore``) or with offset subtraction (like ``OffsetCore``): this is set by ``params.core.bit_sliced.style``. The vector is multiplied by each bit slice matrix separately, digitized, then the results are digitally aggregated using shift-and-add accumulation to implement the power-of-two weighted sum. The parameter ``params.core.bit_sliced.num_slices`` sets the number of bit slices.

Additionally, each of the three cores stores one or more ADC and DAC objects to quantize the MVM/VMM inputs and outputs. These cores also handle the aggregation of input bit slices when input bit slicing is enabled. Note that input bit slicing is independent from weight (or matrix) bit slicing and both types of bit slicing can be simultaneously present. All three core types inherit from the ``WrapperCore`` base class which contains scaling functions for the matrix, inputs, and outputs between algorithmic and hardware units.

These three cores represent different approaches to map numerical values to conductance values. When simulated with ideal devices and the same level of data quantization, the three cores should give the same numerical MVM and VMM results. However, the choice of mapping scheme can have a large yet non-obvious effect on the system's sensitivity to device and circuit level analog errors. This is briefly discussed in Tutorial part 2, and its implications for neural network inference accelerators is extensively explored in Ref. [1].

### NumericCore

The ``NumericCore`` represents a single physical resistive crossbar array and is the interface between the cores and CrossSim's device and array models. The ``set_matrix`` method of this core takes a strictly positive matrix as input, with values in normalized conductance units where a value of 1 represents the maximum device conductance.
- The ``BalancedCore`` uses two ``NumericCore``'s: one for the positive crossbar and one for the negative crossbar.
- The ``OffsetCore`` uses one ``NumericCore``
- The ``BitSlicedCore`` uses _N_ ``NumericCore``'s if using offset subtraction and _2N_ ``NumericCore``'s if using diferential cells, where _N_ is the number of bit slices.

The ``NumericCore`` holds the Device model objects which are used to apply conductance quantization, programming error, drift error, and read noise. It also has selects and accesses the appropriate circuit solver function in ``circuits/array_simulator.py`` to compute the effect of array parasitic resistances on the MVM results.

#### References

[1] T. P. Xiao, B. Feinberg, C. H. Bennett, V. Prabhakar, P. Saxena, V. Agrawal, S. Agarwal, and M. J. Marinella, "On the accuracy of analog neural network inference accelerators," _IEEE Circuits and Systems Magazine_, vol. 22, no. 4, pp. 26-48, 2022.