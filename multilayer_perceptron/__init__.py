"""A multilayer perceptron library built on NumPy."""

from .activation_function import AbstractActivationFunction, get_activation
from .loss_function import AbstractLossFunction, TaskType, get_loss_function
from .network import PerceptronNetwork
from .network_layer import AbstractNetworkLayer, DenseNetworkLayer
from .optimizers import AbstractOptimizer, get_optimizer
from .parameter_initializer import AbstractInitializer, get_initializer

__all__ = [
    "AbstractActivationFunction",
    "AbstractInitializer",
    "AbstractLossFunction",
    "AbstractNetworkLayer",
    "AbstractOptimizer",
    "DenseNetworkLayer",
    "PerceptronNetwork",
    "TaskType",
    "get_activation",
    "get_initializer",
    "get_loss_function",
    "get_optimizer",
]

__version__ = "0.1.0"