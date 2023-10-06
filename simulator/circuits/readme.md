# Circuits

This directory contains various models related to the behavior of the MVM/VMM crossbar array and the peripheral circuitry.

For details on ADCs and DACs, see the READMEs in the ``adc`` and ``dac`` directories.

### Array circuit simulators

In the ideal case, the result of an analog matrix-vector multiplication in a resistive memory array is exactly equivalent to a mathematical matrix-vector multiplication. Among other factors, this equivalence relies on the parasitic metal resistance of the crossbar wires being negligible. The ``array_simulator.py`` file contains circuit solvers that compute the actual result of an analog MVM operation in the presence of metal parasitic resistance. The solver uses parameters in ``params.xbar.array.parasitics``. ``Rp_col`` is the parasitic resistance between two adjacent cells along a column, normalized by ``params.xbar.device.Rmin``. ``Rp_row`` is the equivalent quantity for the rows.

There are several circuit solvers for different assumptions: (1) with current supplied from the input rows/columns and sunk into the output columns/rows, (2) with inputs applied one bit a time to the gates of ideal transistor switches so that parasitics in the input dimension can be ignored, and (3) with cells that sink and source current interleaved along each output column or row to reduce the overall current and parasitic voltage drops. More details can be found in Ref. [1].

All of the circuit solvers have been validated with SPICE simulations, but are much faster due to their use of matrix kernels in Numpy or Cupy. However, the simulation will still be significantly slower than if parasitic resistance is disabled.

#### References

[1] T. P. Xiao, B. Feinberg, J. N. Rohan, C. H. Bennett, S. Agarwal, and M. J Marinella, "Analysis and mitigation of parasitic resistance effects for analog in-memory neural network acceleration", _Semiconductor Science and Technology_, vol. 36, no. 11, 114004, 2021.