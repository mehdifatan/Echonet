Training some state of the art semantic segmentation models for cardiac heart images in pytorch.

The models are as follows:

- Unet
- Linknet
- PSPNet
- FPN
- DeepLabV3

DeepLabV3 with the kernel of resnet50 is one the our models and also there are 4 models of Unet, Linknet, FPN, PSPNet
from segmentation-models-pytorch library.

The main features of this library are:

- High level API (just two lines to create neural network)
- 4 models architectures for binary and multi class segmentation (including legendary Unet)
- 46 available encoders for each architecture
- All encoders have pre-trained weights for faster and better convergence


Installation

PyPI version:

$ pip install segmentation-models-pytorch

Latest version from source:

$ pip install git+https://github.com/qubvel/segmentation_models.pytorch


Each file includes some codes for training and showing testing results in the end of the every epochs.
The best model during the training is saved in the folder of "output" with the train and test loss values
as log file during the traning process. A sample of execution for each file could be as follows:

$ Python segmentation_Unet.py