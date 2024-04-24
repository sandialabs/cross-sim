# CrossSim-PyTorch Integration

This directory contains a work-in-progress implementation of using CrossSim for PyTorch layers. Implemented layers are intended to be drop-in compatible with equivalent PyTorch layers and should have the same behavior when no CrossSim non-idealities are applied.

**As an in-progress feature, there are no guarantees of API stability until this is merged into a main CrossSim release. There may be bugs in this code, please use with caution.**

### Features
- CrossSim Linear layer
- CrossSim Conv2d layer
- All layer types support both analog and digital bias addition
- Layers support gradient computation for CrossSim-in-the-loop training
- Automatic conversion of layers within Torch networks to CrossSim layers

### Known Limitations
- AnalogConv2d layers do not support dilated convolutions.
