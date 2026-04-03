# Array circuit simulators

This directory contains compact circuit models to simulate parasitic resistance and I-V nonlinearity effects in several different array electrical topologies.

### Parasitic resistance solvers

When parasitic resistance is present, the analog matrix-vector multiplication (MVM) operation cannot be simulated as a mathematical MVM with added errors, because parasitic resistance introduces nonlinearity and coupling between memory elements, and the effect is spatially non-uniform across the array. To model these effects accurately, CrossSim internally uses an iterative circuit solver to compute the actual analog computation result. Different circuit solvers are used for different array topologies, as described below, and methods that are shared between these solvers are implemented in the ``IArray`` parent class.

Two critical properties of the array are used to select the corresponding circuit model:
- Interleaved vs. non-interleaved: Whether memory cells for positive and negative weight components in a differential pair are interleaved within an array and connect to the same summation line at the same array position. The positive cell is driven by a positive voltage and the negative cell is driven by a negative voltage, both with respect to the summation column. Interleaving allows local current cancellation along the summation line. If not interleaved, positive and negative components are subtracted at periphery of the array, inside the peripheral circuits, or in the digital domain. If the core style is not ``BALANCED``, this option is irrelevant.
- Input source vs separate source: Input source means that the same interconnect that carries the input signal for the MVM also supplies the current that flows through the memory cells. This input line is orthogonal to the summation line. Separate source means that the current is sourced from a separate interconnect from the input signal, and this line is parallel to the summation line. The input signal is used to connect/disconnect the summation line from the current source line. If using the separate source option, input bit slicing must be used with a slice size of 1.

The four combinations are implemented in the four models: ``NoninterleavedInputSourceArray``, ``NoninterleavedSeparateSourceArray``, ``InterleavedInputSourceArray``, and ``InterleavedSeparateSourceArray``. 

The models can simulate the effect of parasitic resistances of the metal interconnects inside the array (e.g. summation line, input line, source line). It can also simulate series resistance at the terminals of these lines which may be due to peripheral drivers, switches, or muxes. Internally, the solvers use resistance values that are normalized by ``params.xbar.device.Rmin``.  Currently, all of the solvers assume that the terminal of the summation line is held at virtual ground by peripheral circuitry.

If memory device I-V nonlinearity is enabled (``params.xbar.device.nonlinear_IV.enable``) in addition to parasitic resistance, each of the four array models will call the device I-V nonlinearity model to compute the cell currents, given the calculated node voltages in each iteration of the solver.

CrossSim's internal circuit solvers have been validated with SPICE simulations. They are much faster than SPICE, because they are specialized to solving circuit equations for highly regular memory arrays and can exploit Numpy/Cupy's fast matrix kernels. Nonetheless, the simulation will still be slower than if parasitic resistance is disabled.

All the solvers currently make two assumptions: (1) the memory devices can be modeled as linear resistors, (2) the transistor switches in series with the memory device inside the cell can be modeled as ideal switches, i.e. short circuit when on, open circuit when off.

### Simple solver for array of nonlinear I-V devices with no parasitic resistance

The ``UncoupledNonlinearArray`` model is a simple model that is used only when device I-V nonlinearity is enabled but parasitic resistance is disabled, and each summation line is held at a virtual ground. Under these assumptions, the currents through all the devices are uncoupled and an iterative solve is not necessary. This array model calls the device I-V nonlinearity model to compute the cell currents or the summed currents, for the given applied voltages.