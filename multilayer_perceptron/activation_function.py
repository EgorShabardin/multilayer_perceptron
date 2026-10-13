from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .registry import Registry


class AbstractActivationFunction(ABC):
    """Base class for element-wise and row-wise layer activation functions.

    Subclasses implement _forward and _backward. The public forward and
    backward methods handle input conversion and shape checks.

    backward uses values cached by the most recent forward call, so they
    must be called in that order.
    """

    def __init__(self) -> None:
        """Initializes an activation function with an empty cache."""
        self._cache: NDArray[Any] = np.empty(0)
        self._shape: tuple[int, ...] | None = None

    @property
    def gain(self) -> float:
        """Recommended gain for initializing the weights of the preceding layer."""
        return 1.0

    def forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies the activation function.

        Args:
            pre_activation (NDArray[np.float64]): Layer output before activation.

        Returns:
            NDArray[np.float64]: Activated values of the same shape.
        """
        pre_activation = np.asarray(pre_activation, dtype=np.float64)
        self._shape = pre_activation.shape
        return self._forward(pre_activation)
    
    def backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Propagates the gradient through the activation function.

        Args:
            output_gradient (NDArray[np.float64]): Gradient of the loss with
                respect to the activation output.

        Returns:
            NDArray[np.float64]: Gradient of the loss with respect to the
                activation input.

        Raises:
            RuntimeError: If called before forward.
            ValueError: If the gradient shape differs from the last input shape.
        """
        if self._shape is None:
            raise RuntimeError("backward() was called before forward()")
        output_gradient = np.asarray(output_gradient, dtype=np.float64)
    
        if output_gradient.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {output_gradient.shape}")
        return self._backward(output_gradient)

    @abstractmethod
    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the activation and caches what _backward needs."""

    @abstractmethod
    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the input gradient from the output gradient and the cache."""


_ACTIVATION_REGISTRY: Registry[AbstractActivationFunction] = Registry("activations")


@_ACTIVATION_REGISTRY.register(registration_name="identity")
class Identity(AbstractActivationFunction):
    """Identity activation: f(x) = x. Gain is 1."""

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        return pre_activation.copy()

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return output_gradient.copy()


@_ACTIVATION_REGISTRY.register(registration_name="relu")
class ReLU(AbstractActivationFunction):
    """ReLU: f(x) = max(x, 0). Gain is sqrt(2)."""

    @property
    def gain(self) -> float:
        return np.sqrt(2.0)

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        self._cache = pre_activation > 0.0
        return np.maximum(pre_activation, 0.0)

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(self._cache, output_gradient, 0.0)


@_ACTIVATION_REGISTRY.register(registration_name="leaky_relu")
class LeakyReLU(AbstractActivationFunction):
    """Leaky ReLU: f(x) = x if x > 0 else slope * x.

    Gain is sqrt(2 / (1 + slope^2)).
    """
    
    def __init__(self, slope: float) -> None:
        """Initializes Leaky ReLU.

        Args:
            slope (float): Slope for negative inputs, in the open interval (0, 1).

        Raises:
            ValueError: If slope is not in (0, 1).
        """
        if not 0.0 < slope < 1.0:
            raise ValueError(f"Slope must be in the open interval (0, 1), got {slope!r}")

        super().__init__()
        self._slope: float = float(slope)

    @property
    def gain(self) -> float:
        return np.sqrt(2.0 / (1.0 + self._slope * self._slope))

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        self._cache = pre_activation > 0.0
        return np.maximum(pre_activation, self._slope * pre_activation)

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(self._cache, output_gradient, self._slope * output_gradient)


@_ACTIVATION_REGISTRY.register(registration_name="sigmoid")
class Sigmoid(AbstractActivationFunction):
    """Sigmoid: f(x) = 1 / (1 + exp(-x)). Gain is 1."""

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        self._cache = 1.0 / (1.0 + np.exp(-np.clip(pre_activation, -500.0, 500.0)))
        return self._cache.copy()

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return output_gradient * self._cache * (1.0 - self._cache)


@_ACTIVATION_REGISTRY.register(registration_name="tanh")
class Tanh(AbstractActivationFunction):
    """Hyperbolic tangent. Gain is 5/3."""

    @property
    def gain(self) -> float:
        return 5.0 / 3.0

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        self._cache = np.tanh(pre_activation)
        return self._cache.copy()

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return output_gradient * (1.0 - self._cache * self._cache)


@_ACTIVATION_REGISTRY.register(registration_name="softmax")
class Softmax(AbstractActivationFunction):
    """Softmax over the last axis. Gain is 1."""

    def _forward(self, pre_activation: NDArray[np.float64]) -> NDArray[np.float64]:
        exp_value = np.exp(pre_activation - np.max(pre_activation, axis=-1, keepdims=True))
        self._cache = exp_value / np.sum(exp_value, axis=-1, keepdims=True)
        return self._cache.copy()

    def _backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._cache * (output_gradient - np.sum(output_gradient * self._cache, axis=-1, keepdims=True))


def get_activation(activation_name: str, **kwargs: Any) -> AbstractActivationFunction:
    """Creates an activation function by its registered name.

    Args:
        activation_name (str): Registered activation name.
        **kwargs (Any): Extra keyword arguments for the activation.

    Returns:
        AbstractActivationFunction: A new activation function instance.

    Raises:
        KeyError: If no activation is registered under activation_name.
        ValueError: If activation_name is empty or the arguments are invalid.
    """
    return _ACTIVATION_REGISTRY.create(activation_name, **kwargs)