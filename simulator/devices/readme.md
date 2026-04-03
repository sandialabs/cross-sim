# Devices

This directory contains models for the resistive memory device used to compute the individual analog multiplications in a crossbar MVM or VMM. These models are mainly selected and controlled by parameters inside the ``DeviceParameters`` class (``params.xbar.device``, passed into the class as ``device_params``).

Device models are implemented as classes that inherit from the ``IDevice`` class. The ``IDevice`` class natively implements the quantization of the conductance levels that can be targeted for programming. The resolution is set by the parameter ``params.xbar.device.cell_bits``.

Devices can also implement models for programming error, conductance drift, read noise, and I-V nonlinearity.

#### Programming error
Programming error is specified by the ``programming_error`` function which can be user-defined. This function takes the target conductance matrix (after quantization) as input and returns a conductance matrix with programming errors applied. Errors can be have a combination of random and deterministic components and can be state-dependent. These errors are permanently applied to the arrays that are stored in a core, until a new matrix is programmed or the core is destroyed.

#### Conductance drift
Conductance drift is specified by the ``drift_error`` function which can be user-defined. This function is similar to programming error, but also takes a time variable (``device_params.time``) as parameter, which should be interpreted as the time elapsed after programming. If time is non-zero, the drift model is applied and programming error is ignored. This is based on the assumption that measurements of conductance taken some time after programming take into account both the initial programming errors and the subsequent drift. In the current CrossSim version, time is fixed during a full simulation. In a future update, we will add the ability to evolve time during a simulation and accumulate errors over time.

#### Read noise
Read noise is specified by the ``read_noise`` function which can be user-defined. The function takes as input the conductance matrix with programming or drift errors applied, and returns a conductance matrix with read noise added. The read noise is re-applied each time an MVM/VMM is called on a core, and the conductance change due to noise is not persistent. This simulates white noise with no temporal correlations. The properties of noise can be state-dependent.

#### I-V nonlinearity
I-V nonlinearity is specified by two functions: ``nonlinear_current`` and ``nonlinear_current_sum``. These two functions model the same effect, and only one of the two will be used, based on the parameter ``params.simulation.fast_nonlinear_IV``. It is not necessary to implement both functions to model I-V nonlinearity, but it is recommended to at least implement ``nonlinear_current``. The I-V nonlinearity model is fully user-specified: any arbitrary shape of the I-V curve can be modeled and the I-V curve can be state-dependent. If I-V nonlinearity is enabled, the nonlinear current function will be called every time an MVM or VMM is called on a core (like ``read_noise``). If both functions are implemented, the user should ensure that they are physically consistent.

``nonlinear_current`` takes as input a matrix of conductances and a matrix of applied voltages _at every cell of the array_, both in normalized units. The function returns the current at every cell of the array. This function is used when `params.simulation.fast_nonlinear_IV = False`. It is important to note that since we are modeling nonlinear I-V, the conductance is mainly used to distinguish the memory states of a device: it is not necessarily the conductance of the device at the applied voltage. The exact meaning of the conductance is up to the user: it depends on how the nonlinear I-V model is implemented. For example, in the I-V nonlinearity model for the ``RRAMMilo`` device, the conductance refers to the conductance of the device at zero applied voltage.

``nonlinear_current_sum`` takes as input a matrix of conductances and vector (or matrix) of applied voltages _to the terminals of the array_, both in normalized units. This function is used when ``params.simulation.fast_nonlinear_IV = True``. This function returns the _summed currents_ in every summation line of the array. It simulates the analog MVM or VMM without requiring the currents in all the individual cells to be stored, which saves time and memory. This function can be used when all the cells that share the same input also have the same applied voltage, which is true only when there are zero parasitic IR drops.


### Selecting a device model

A device class can implement a model for any, all, or none of the error properties above. When setting device properties during a simulation, the model for each type of error can be independently specified. This allows selective disables of some error models, or allows error models to be mix-and-matched to create a Frankenstein device. For example, one possibility is:
```
params.xbar.device.read_noise.model = "IdealDevice"
params.xbar.device.programming_error.model = "NormalProportionalDevice"
params.xbar.device.drift_error.model = "SONOS"
params.xbar.device.nonlinear_IV.model = "RRAMMilo"
```
Each model above also has a ``enable`` parameter which if ``False``, is the same as setting the model to ``IdealDevice``.


### Available device models

The ``IdealDevice`` class implements a device that has no errors other than quantization, so that the device conductances are not perturbed relative to their target values.

There are generic device models that implement parameterizable device models that do not represent any specific device. This can be used to perform a sweep of the device precision. There are several generic device models, whose names are of the form: ``(X)(Y)Device``. These are implemented in ``generic_error.py``.
- ``(X)`` is either ``Normal`` or ``Uniform``. This specifies whether the random conductance error that is applied is drawn from a normal distribution or a uniform distribution. The width of the distribution is controlled by the parameter ``params.xbar.device.programming_error.magnitude`` or ``params.xbar.device.read_noise.magnitude``.
- ``(Y)`` is either ``Independent``, ``Proportional``, or ``InverseProportional``. The first means the magnitude of the random error is independent of the conductance state, the second means the magnitude is proportional to the conductance state, and the third means the magnitude is proportional to the resistance state.

Finally, three models are available in the ``custom`` directory that correspond to actual measured devices in the literature. These are: ``SONOS`` [1], ``PCMJoshi`` [2], and ``RRAMMilo`` [3]. For more details, see the corresponding files.

### Adding a new custom device model

To add a new device model, follow these steps:

- Create a new file in the ``custom`` directory, and create a class that inherits from ``EmptyDevice``. See the existing files in the ``custom`` directory for examples of the appropriate imports and constructor behavior.
- Write functions for programming error, drift error, and/or read noise. Any behavior that can be specified as a Python function can be implemented. Note that all three functions take as input a normalized conductance matrix where the maximum target-able conductance is 1. The output must also be in these normalized conductance units. For examples, see the existing files in the ``custom`` directory.
- At the top of ``device.py``, import the created device class from the new file.

You can now set the device model to the name of the created device class, by setting the appropriate ``model`` parameter as shown above.

It is also possible to make custom parameterizable device models. In this case, make sure to add these as new parameters accessible from ``params.device.xbar``, so that they can be accessed from the device model. See the ``parameters`` README for details on how to do this.

 
#### References

[1] V. Agrawal, T. P. Xiao, C. H. Bennett, B. Feinberg, S. Shetty, K. Ramkumar, H. Medu, K. Thekkekara, R. Chettuvetty, S. Leshner, Z. Luzada, L. Hinh, T. Phan, M. J. Marinella, and S. Agarwal, "Subthreshold operation of SONOS analog memory to enable accurate low-power neural network inference", _IEEE International Electron Devices Meeting (IEDM)_, 21.7.1-21.7.4, 2022.

[2] V. Joshi, M. Le Gallo, S. Haefeli, I. Boybat, S. R. Nandakumar, C. Piveteau, M. Dazzi, B. Rajendran, A. Sebastian, and E. Eleftheriou, "Accurate deep neural network inference using computational phase-change memory", _Nature Communications_, 11, 2473, 2020.

[3] V. Milo, F. Anzalone, C. Zambelli, E. Perez, M. K. Mahadevaiah, O. G. Ossorio, P. Olivo, C. Wenger, and D. Ielmini, "Optimized programming algorithms for multilevel RRAM in hardware neural networks," _IEEE International Reliability Physics Symposium (IRPS)_, pp. 1-6, 2021.