from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .registry import Registry


class AbstractInitializer(ABC):
    """Base class for weight and bias initializers of a linear layer.

    The weight matrix has shape (in_features, out_features), i.e. the layer
    computes y = x @ W + b. Biases are always initialized with zeros.
    """

    def __init__(self, in_features: int, out_features: int, gain: float) -> None:
        """Initializes the initializer with the layer dimensions.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            gain (float): Scaling factor for the weights, usually provided
                by the activation function that follows the layer.

        Raises:
            ValueError: If in_features or out_features is not a positive
                integer, or gain is not a positive finite number.
        """
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError(f"gain must be a positive finite number, got {gain!r}")
        
        self._in_features: int = in_features
        self._out_features: int = out_features
        self._gain: float = float(gain)

    def init_parameters(self, rng: np.random.Generator | int | None = None) -> dict[str, NDArray[np.float64]]:
        """Initializes the parameters.

        Args:
            rng (np.random.Generator | int | None): A random number generator,
                a seed, or None for a fresh unseeded generator.

        Returns:
            dict[str, NDArray[np.float64]]: "weight" of shape (in_features, out_features)
                and "bias" of shape (out_features,).
        """
        rng = np.random.default_rng(rng)
        return {"weight": self._init_weight(rng), "bias": np.zeros(self._out_features, dtype=np.float64)}

    @abstractmethod
    def _init_weight(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Creates the weight matrix.

        Args:
            rng (np.random.Generator): Random number generator.

        Returns:
            NDArray[np.float64]: Weight matrix of shape (in_features, out_features).
        """


_INITIALIZER_REGISTRY: Registry[AbstractInitializer] = Registry("initializers")


@_INITIALIZER_REGISTRY.register(registration_name="xavier_normal")
class XavierNormal(AbstractInitializer):
    """Xavier (Glorot) normal initializer.

    Samples from N(0, std^2) with std = gain * sqrt(2.0 / (in_features + out_features)).
    """

    def _init_weight(self, rng: np.random.Generator) -> NDArray[np.float64]:
        std: float = self._gain * np.sqrt(2.0 / (self._in_features + self._out_features))
        return rng.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))


@_INITIALIZER_REGISTRY.register(registration_name="xavier_uniform")
class XavierUniform(AbstractInitializer):
    """Xavier (Glorot) uniform initializer.

    Samples from U(-limit, limit) with limit = gain * sqrt(6.0 / (in_features + out_features)).
    """

    def _init_weight(self, rng: np.random.Generator) -> NDArray[np.float64]:
        limit: float = self._gain * np.sqrt(6.0 / (self._in_features + self._out_features))
        return rng.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))


@_INITIALIZER_REGISTRY.register(registration_name="kaiming_normal")
class KaimingNormal(AbstractInitializer):
    """Kaiming (He) normal initializer.

    Samples from N(0, std^2) with std = gain / sqrt(in_features).
    """

    def _init_weight(self, rng: np.random.Generator) -> NDArray[np.float64]:
        std: float = self._gain / np.sqrt(self._in_features)
        return rng.normal(loc=0.0, scale=std, size=(self._in_features, self._out_features))


@_INITIALIZER_REGISTRY.register(registration_name="kaiming_uniform")
class KaimingUniform(AbstractInitializer):
    """Kaiming (He) uniform initializer.

    Samples from U(-limit, limit) with limit = gain * sqrt(3.0 / in_features).
    """

    def _init_weight(self, rng: np.random.Generator) -> NDArray[np.float64]:
        limit: float = self._gain * np.sqrt(3.0 / self._in_features)
        return rng.uniform(low=-limit, high=limit, size=(self._in_features, self._out_features))


def get_initializer(initialization_name: str, in_features: int, out_features: int, gain: float) -> AbstractInitializer:
    """Creates an initializer by its registered name.

    Args:
        initialization_name (str): Registered initializer name.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        gain (float): Scaling factor for the weights, usually provided
            by the activation function that follows the layer.

    Returns:
        AbstractInitializer: A new initializer instance.

    Raises:
        KeyError: If no initializer is registered under initialization_name.
        ValueError: If initialization_name is empty, the dimensions are
            invalid, or gain is not a positive finite number.
    """
    return _INITIALIZER_REGISTRY.create(initialization_name, in_features, out_features, gain)