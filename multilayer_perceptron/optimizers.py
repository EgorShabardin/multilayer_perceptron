from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .network_layer import AbstractNetworkLayer
from .registry import Registry


class AbstractOptimizer(ABC):
    """Base class for optimizers.

    Subclasses implement update_parameters, which performs one optimization
    step using the gradients stored by the most recent backward call of each layer.
    """

    @abstractmethod
    def update_parameters(self, layers_lst: list[AbstractNetworkLayer]) -> None:
        """Performs one optimization step.

        Args:
            layers_lst (list[AbstractNetworkLayer]): Layers whose parameters are
                updated in place using their current gradients.
        """


_OPTIMIZER_REGISTRY: Registry[AbstractOptimizer] = Registry("Optimizers")


@_OPTIMIZER_REGISTRY.register(registration_name="sgd")
class SGD(AbstractOptimizer):
    """Stochastic gradient descent.

    Update rule: parameter = parameter - learning_rate * gradient.
    """

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initializes stochastic gradient descent.

        Args:
            learning_rate (float): The step size. Must be a positive finite number.

        Raises:
            ValueError: If learning_rate is not a positive finite number.
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate!r}")
        self._learning_rate: float = float(learning_rate)

    def update_parameters(self, layers_lst: list[AbstractNetworkLayer]) -> None:
        """Moves each parameter against its gradient.

        Args:
            layers_lst (list[AbstractNetworkLayer]): Layers whose parameters are
                updated in place using their current gradients.
        """
        for layer in layers_lst:
            for name, parameter in layer.parameters.items():
                parameter -= self._learning_rate * layer.gradients[name]


@_OPTIMIZER_REGISTRY.register(registration_name="momentum")
class Momentum(AbstractOptimizer):
    """Stochastic gradient descent with momentum.

    Update rule: velocity = beta * velocity + gradient,
        parameter = parameter - learning_rate * velocity.
    """

    def __init__(self, learning_rate: float = 1e-3, beta: float = 0.9) -> None:
        """Initializes stochastic gradient descent with momentum.

        Args:
            learning_rate (float): The step size. Must be a positive finite number.
            beta (float): Velocity decay rate, in the interval [0, 1).

        Raises:
            ValueError: If learning_rate is not a positive finite number or beta is not in [0, 1).
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate!r}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in the interval [0, 1), got {beta!r}")
        self._learning_rate: float = float(learning_rate)
        self._beta: float = float(beta)
        self._velocity: dict[AbstractNetworkLayer, dict[str, NDArray[np.float64]]] = {}

    def update_parameters(self, layers_lst: list[AbstractNetworkLayer]) -> None:
        """Updates the velocity of each parameter and moves the parameter.

        Velocities are created with zeros the first time a layer or
            parameter is seen.

        Args:
            layers_lst (list[AbstractNetworkLayer]): Layers whose parameters are
                updated in place using their current gradients.
        """
        for layer in layers_lst:
            layer_velocity: dict[str, NDArray[np.float64]] = self._velocity.setdefault(layer, {})
            for name, parameter in layer.parameters.items():
                velocity: NDArray[np.float64] = layer_velocity.setdefault(name, np.zeros_like(parameter))
                velocity *= self._beta
                velocity += layer.gradients[name]
                parameter -= self._learning_rate * velocity


@_OPTIMIZER_REGISTRY.register(registration_name="adam")
class Adam(AbstractOptimizer):
    """Adam optimizer.

    Update rule: first_moment = first_beta * first_moment + (1 - first_beta) * gradient,
        second_moment = second_beta * second_moment + (1 - second_beta) * gradient^2,
        first_moment_hat = first_moment / (1 - first_beta^t),
        second_moment_hat = second_moment / (1 - second_beta^t),
        parameter = parameter - learning_rate * first_moment_hat / (sqrt(second_moment_hat) + eps).
    """

    def __init__(self, learning_rate: float = 1e-3, first_beta: float = 0.9,
                 second_beta: float = 0.999, eps: float = 1e-8) -> None:
        """Initializes the Adam optimizer.

        Args:
            learning_rate (float): The step size. Must be a positive finite number.
            first_beta (float): Decay rate of the first moment, in the interval [0, 1).
            second_beta (float): Decay rate of the second moment, in the interval [0, 1).
            eps (float): Small positive constant added to the denominator for numerical stability.

        Raises:
            ValueError: If learning_rate or eps is not a positive finite
                number, or first_beta or second_beta is not in [0, 1).
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be a positive finite number, got {learning_rate!r}")
        if not 0.0 <= first_beta < 1.0:
            raise ValueError(f"first_beta must be in the interval [0, 1), got {first_beta!r}")
        if not 0.0 <= second_beta < 1.0:
            raise ValueError(f"second_beta must be in the interval [0, 1), got {second_beta!r}")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be a positive finite number, got {eps!r}") 
        self._learning_rate: float = float(learning_rate)
        self._first_beta: float = float(first_beta)
        self._second_beta: float = float(second_beta)
        self._eps: float = float(eps)

        self._step: int = 0
        self._first_moment: dict[AbstractNetworkLayer, dict[str, NDArray[np.float64]]] = {}
        self._second_moment: dict[AbstractNetworkLayer, dict[str, NDArray[np.float64]]] = {}

    def update_parameters(self, layers_lst: list[AbstractNetworkLayer]) -> None:
        """Updates both moments of each parameter and moves the parameter.

        Increments the step counter once per call, then applies the bias
            corrected Adam update. Moments are created with zeros the first
            time a layer or parameter is seen.

        Args:
            layers_lst (list[AbstractNetworkLayer]): Layers whose parameters are
                updated in place using their current gradients.
        """
        self._step += 1
        first_correction: float = 1.0 - self._first_beta ** self._step
        second_correction: float = 1.0 - self._second_beta ** self._step

        for layer in layers_lst:
            first_layer_moment: dict[str, NDArray[np.float64]] = \
                self._first_moment.setdefault(layer, {})
            second_layer_moment: dict[str, NDArray[np.float64]] = \
                self._second_moment.setdefault(layer, {})
            for name, parameter in layer.parameters.items():
                gradient: NDArray[np.float64] = layer.gradients[name]
                first_moment: NDArray[np.float64] = \
                    first_layer_moment.setdefault(name, np.zeros_like(parameter))
                second_moment: NDArray[np.float64] = \
                    second_layer_moment.setdefault(name, np.zeros_like(parameter))

                first_moment *= self._first_beta
                first_moment += (1.0 - self._first_beta) * gradient
                second_moment *= self._second_beta
                second_moment += (1.0 - self._second_beta) * gradient * gradient

                first_moment_hat: NDArray[np.float64] = first_moment / first_correction
                second_moment_hat: NDArray[np.float64] = second_moment / second_correction
                parameter -= self._learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + self._eps)


def get_optimizer(optimizer_name: str, **kwargs: Any) -> AbstractOptimizer:
    """Creates an optimizer by its registered name.

    Args:
        optimizer_name (str): Registered optimizer name.
        **kwargs (Any): Extra keyword arguments for the optimizer.

    Returns:
        AbstractOptimizer: A new optimizer instance.

    Raises:
        KeyError: If no optimizer is registered under optimizer_name.
        ValueError: If optimizer_name is empty or the arguments are invalid.
    """
    return _OPTIMIZER_REGISTRY.create(optimizer_name, **kwargs)
