"""A multilayer perceptron library built on NumPy."""

from .parameter_initializer import AbstractInitializer, get_initializer
from .activation_function import AbstractActivationFunction, get_activation
from .loss_function import AbstractLossFunction, TaskType, get_loss_function
from .optimizers import AbstractOptimizer, get_optimizer
from .network_layer import AbstractNetworkLayer, DenseNetworkLayer
from .network import PerceptronNetwork

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