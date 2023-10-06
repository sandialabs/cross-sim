# Parameters

The files in this directory implement CrossSim's parameters infrastructure for setting the behavior of cores and configuring the models of devices and circuits.

All of the parameters used in a CrossSim simulation are consolidated in a ``CrossSimParameters`` object. This object has three attributes which splits the parameters into three broad categories:
- Core parameters (``CoreParameters``) configure how data such as matrix weights and vector inputs are mapped to hardware, including the range of numerical values that can be represented.
- Crossbar parameters (``XbarParameters``) configure the properties and models used for the resistive devices, the memory array's electrical topology and parasitics, and the properties of the peripheral circuits, most notably the analog-to-digital converters (ADCs) and the digital-to-analog converters (DACs).
- Simulation parameters (``SimulationParameters``) contain settings related to the numerical simulation, rather than how the analog hardware is modeled. These include parameters related to GPU usage, convergence of the circuit solver, data profiling, and performance.

Each category of parameters above can have their own sub-categories of parameters as attributes; this forms a hierarchy of parameters. Here is an example where we set a specific parameter:
```
params = CrossSimParameters()
params.xbar.array.parasitics.Rp_col = 1
```
In this example,
- ``params`` is a ``CrossSimParameters`` object
- ``params.xbar`` is an ``XbarParameters`` object
- ``params.xbar.array`` groups together parameters related to the electrical properties of the memory array
- ``params.xbar.array.parasitics`` groups together parameters related to array parasitic resistance
- ``params.xbar.array.parasitics.Rp_col`` is the parasitic metal resistance (in Ohms) between two unit cells along an array column

Please see ``applications/mvm_params.py`` for an example script that sets most of the parameters used by CrossSim. If a parameter is not explicitly set, its value will be the default value defined inside the relevant parameters class.

All parameter objects inherit from the ``BaseParameters`` class which dictates their structure and implements an interface to import from or export parameters to JSON files.

Parameters objects can inherit from each other. For example, parameters related to a custom ADC model (e.g. ``RampADCParameters``, ``SarADCParameters``) should inherit from the ``ADCParameters`` class which contains parameters that should be common to all ADC implementations, such as the number of bits of resolution.

To add a new parameter to an existing group of parameters, first find the relevant class in ``core_parameters.py``, ``xbar_parameters.py`` or ``simulation_parameters.py``. Then add the new parameter as an attribute and set its default value inside the class definition.

To add a new group of parameters, define a new class (e.g. ``MyParameters``) for that group that, if applicable, inherits from an existing class of parameters. If there is no such existing class, inherit from ``BaseParameters``. Then define and document the new parameters as attributes of ``MyParameters``, and set their default values. Finally, the new group of parameters must be accessible as an attribute of an existing class that serves as a super-category of parameters: add an attribute to that class with the type set to ``MyParameters``.