from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .activation_function import AbstractActivationFunction, get_activation
from .parameter_initializer import AbstractInitializer, get_initializer


class AbstractNetworkLayer(ABC):
    """Base class for network layers.

    A layer owns its trainable parameters and the gradients.
    """

    def __init__(self) -> None:
        """Initializes a layer with no parameters."""
        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._gradients: dict[str, NDArray[np.float64]] = {}

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        """Trainable parameters."""
        return self._parameters

    @property
    def gradients(self) -> dict[str, NDArray[np.float64]]:
        """Gradients of the loss with respect to each parameter."""
        return self._gradients

    def set_parameters(self, parameters: dict[str, NDArray[np.float64]]) -> None:
        """Copies parameter values into the layer.

        Args:
            parameters (dict[str, NDArray[np.float64]]): New values by name.

        Raises:
            KeyError: If a name is not a parameter of this layer.
            ValueError: If a value shape differs from the parameter shape.
        """
        converted: dict[str, NDArray[np.float64]] = {}
        for name, value in parameters.items():
            if name not in self._parameters:
                expected_names: str = ", ".join(sorted(self._parameters))
                raise KeyError(f"Unknown parameter '{name}'. Expected one of: {expected_names}")
            array: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
            expected_shape: tuple[int, ...] = self._parameters[name].shape
            if array.shape != expected_shape:
                raise ValueError(f"Parameter '{name}' must have shape {expected_shape}, got {array.shape}")
            converted[name] = array

        for name, array in converted.items():
            self._parameters[name][...] = array

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Constructor arguments needed to recreate this layer."""
    
    @abstractmethod
    def forward(self, input_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the layer output and caches what backward needs."""

    @abstractmethod
    def backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Stores parameter gradients and returns the gradient for the input."""


class DenseNetworkLayer(AbstractNetworkLayer):
    """Fully connected layer: y = activation(x @ W + b)."""

    def __init__(self, in_features: int, out_features: int, activation_name: str,
                 initialization_name: str, *, rng: np.random.Generator | int | None = None,
                 activation_kwargs: dict[str, Any] | None = None) -> None:
        """Initializes the layer and its parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            activation_name (str): Registered activation name.
            initialization_name (str): Registered initializer name.
            rng (np.random.Generator | int | None): Random number generator
                or seed for weight initialization.
            activation_kwargs (dict[str, Any] | None): Extra keyword arguments
                for the activation.

        Raises:
            KeyError: If the activation or initializer name is not registered.
            ValueError: If a dimension is not a positive integer or the activation
                arguments are invalid.
        """
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        
        self._in_features: int = in_features
        self._out_features: int = out_features
        self._activation_name: str = activation_name
        self._initialization_name: str = initialization_name
        self._activation_kwargs: dict[str, Any] = dict(activation_kwargs or {})

        self._activation_function: AbstractActivationFunction = \
            get_activation(activation_name, **(activation_kwargs or {}))
        self._initializer: AbstractInitializer = \
            get_initializer(initialization_name, in_features, out_features, self._activation_function.gain)

        self._input_matrix: NDArray[np.float64] | None = None
        self._parameters = self._initializer.init_parameters(rng)
        self._gradients = {name: np.zeros_like(value) for name, value in self._parameters.items()}

    def forward(self, input_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes activation(x @ W + b).

        Args:
            input_matrix (NDArray[np.float64]): Activation of the previous layer.

        Returns:
            NDArray[np.float64]: Activation of the current layer.

        Raises:
            ValueError: If the input is the wrong size.
        """
        input_matrix = np.asarray(input_matrix, dtype=np.float64)
        if input_matrix.ndim != 2 or input_matrix.shape[1] != self._in_features:
            raise ValueError(f"Expected input of shape (batch, {self._in_features}), got {input_matrix.shape}")
        self._input_matrix = input_matrix
        pre_activation: NDArray[np.float64] = input_matrix @ self._parameters["weight"] + self._parameters["bias"]
        return self._activation_function.forward(pre_activation)

    def backward(self, output_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes parameter gradients and the gradient for the input.

        Args:
            output_gradient (NDArray[np.float64]): Gradient of the loss with
                respect to the layer output.

        Returns:
            NDArray[np.float64]: Gradient of the loss with respect to the layer input.

        Raises:
            RuntimeError: If called before forward.
            ValueError: If the gradient shape differs from the output shape.
        """
        if self._input_matrix is None:
            raise RuntimeError("backward() was called before forward()")
        delta: NDArray[np.float64] = self._activation_function.backward(output_gradient)
        self._gradients["weight"][...] = self._input_matrix.T @ delta
        self._gradients["bias"][...] = np.sum(delta, axis=0)
        return delta @ self._parameters["weight"].T

    @property
    def config(self) -> dict[str, Any]:
        """Constructor arguments needed to recreate this layer."""
        return {
            "in_features": self._in_features,
            "out_features": self._out_features,
            "activation_name": self._activation_name,
            "initialization_name": self._initialization_name,
            "activation_kwargs": dict(self._activation_kwargs)
        }
