from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .registry import Registry


class AbstractLossFunction(ABC):
    """Base class for loss functions.

    Forward computes a scalar loss and caches what backward needs.
    Backward returns the gradient of that loss with respect to the
    prediction, so it must be called after forward.
    """

    def __init__(self) -> None:
        """Initializes a loss function with an empty cache."""
        self._cache: NDArray[np.float64] = np.empty(0)
        self._forward_called: bool = False

    def forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        """Computes the loss.

        Args:
            prediction (NDArray[np.float64]): Model output.
            target (NDArray[np.float64]): Ground truth of the same shape.

        Returns:
            float: The scalar loss value.

        Raises:
            ValueError: If prediction and target shapes differ.
        """
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: prediction {prediction.shape}, target {target.shape}")
        
        loss = self._forward(prediction, target)
        self._forward_called = True
        return loss
    
    def backward(self) -> NDArray[np.float64]:
        """Returns the gradient of the last computed loss.

        Returns:
            NDArray[np.float64]: Gradient with respect to the prediction.

        Raises:
            RuntimeError: If called before forward.
        """
        if not self._forward_called:
            raise RuntimeError("backward() was called before forward()")
        return self._backward()

    @abstractmethod
    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        """Computes the loss and caches what _backward needs."""

    @abstractmethod
    def _backward(self) -> NDArray[np.float64]:
        """Computes the gradient from the cache."""


_LOSS_REGISTRY: Registry[AbstractLossFunction] = Registry("Losses")


@_LOSS_REGISTRY.register(registration_name="mse")
class MeanSquaredError(AbstractLossFunction):
    """Mean squared error: L = mean((p - y)^2).

    The mean is taken over all elements. Gradient: 2 * (p - y) / size.
    """

    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        self._cache = prediction - target
        return float(np.mean(self._cache * self._cache))

    def _backward(self) -> NDArray[np.float64]:
        return 2.0 * self._cache / self._cache.size


@_LOSS_REGISTRY.register(registration_name="mae")
class MeanAbsoluteError(AbstractLossFunction):
    """Mean absolute error: L = mean(|p - y|).

    The mean is taken over all elements. Gradient: sign(p - y) / size.
    """

    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        self._cache = prediction - target
        return float(np.mean(np.abs(self._cache)))

    def _backward(self) -> NDArray[np.float64]:
        return np.sign(self._cache) / self._cache.size


@_LOSS_REGISTRY.register(registration_name="bce")
class BinaryCrossEntropy(AbstractLossFunction):
    """Binary cross-entropy on probabilities.

    L = -mean(y * log(p) + (1 - y) * log(1 - p)), averaged over all elements.
    Gradient: ((p - y) / (p * (1 - p))) / size.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initializes binary cross-entropy.

        Args:
            eps (float): Probabilities are clipped to [eps, 1 - eps].

        Raises:
            ValueError: If eps is not in (0, 0.5).
        """
        if not 0.0 < eps < 0.5:
            raise ValueError(f"eps must be in the open interval (0, 0.5), got {eps!r}")
        super().__init__()
        self._eps: float = float(eps)

    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        prediction = np.clip(prediction, self._eps, 1.0 - self._eps)
        self._cache = (prediction - target) / (prediction * (1.0 - prediction))
        return float(-np.mean(target * np.log(prediction) + (1 - target) * np.log1p(-prediction)))

    def _backward(self) -> NDArray[np.float64]:
        return self._cache / self._cache.size


@_LOSS_REGISTRY.register(registration_name="bce_with_logits")
class BinaryCrossEntropyWithLogits(AbstractLossFunction):
    """Binary cross-entropy on raw logits.

    L = mean(max(z, 0) - z * y + log(1 + exp(-|z|))), averaged over all elements.
    Gradient: (sigmoid(z) - y) / size. Use Identity as the last activation.
    """

    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        sigmoid: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-np.clip(prediction, -500.0, 500.0)))
        self._cache = sigmoid - target
        return float(np.mean(np.maximum(prediction, 0.0) - prediction * target +
                             np.log1p(np.exp(-np.abs(prediction)))))

    def _backward(self) -> NDArray[np.float64]:
        return self._cache / self._cache.size


@_LOSS_REGISTRY.register(registration_name="cce")
class CategoricalCrossEntropy(AbstractLossFunction):
    """Categorical cross-entropy on probabilities.

    L = -mean(sum(y * log(p), axis=-1)). Expects one-hot targets and p that
    sums to 1, e.g. after Softmax. Gradient: -y / p / n_samples
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initializes categorical cross-entropy.

        Args:
            eps (float): Probabilities are clipped to [eps, 1 - eps].

        Raises:
            ValueError: If eps is not in (0, 0.5).
        """
        if not 0.0 < eps < 0.5:
            raise ValueError(f"eps must be in the open interval (0, 0.5), got {eps!r}")
        super().__init__()
        self._eps: float = float(eps)
    
    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        prediction = np.clip(prediction, self._eps, 1.0 - self._eps)
        self._cache = -target / prediction
        return float(-np.mean(np.sum(target * np.log(prediction), axis=-1)))

    def _backward(self) -> NDArray[np.float64]:
        n_samples: int = self._cache.size // self._cache.shape[-1]
        return self._cache / n_samples


@_LOSS_REGISTRY.register(registration_name="cce_with_logits")
class CategoricalCrossEntropyWithLogits(AbstractLossFunction):
    """Categorical cross-entropy on raw logits.

    L = -mean(sum(y * log_softmax, axis=-1)).
    Gradient: (softmax(z) - y) / n_samples. Use Identity as the last activation.
    """
    
    def _forward(self, prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        shifted: NDArray[np.float64] = prediction - np.max(prediction, axis=-1, keepdims=True)
        log_softmax: NDArray[np.float64] = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        self._cache = np.exp(log_softmax) - target
        return float(-np.mean(np.sum(target * log_softmax, axis=-1)))

    def _backward(self) -> NDArray[np.float64]:
        n_samples: int = self._cache.size // self._cache.shape[-1]
        return self._cache / n_samples


def get_loss_function(loss_name: str, **kwargs: Any) -> AbstractLossFunction:
    """Creates a loss function by its registered name.

    Args:
        loss_name (str): Registered loss name.
        **kwargs (Any): Extra keyword arguments for the loss.

    Returns:
        AbstractLossFunction: A new loss function instance.

    Raises:
        KeyError: If no loss is registered under loss_name.
        ValueError: If loss_name is empty or the arguments are invalid.
    """
    return _LOSS_REGISTRY.create(loss_name, **kwargs)
