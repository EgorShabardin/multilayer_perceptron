import json
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from rich import print as rich_print
from rich.progress import track

from .loss_function import AbstractLossFunction, TaskType, get_loss_function
from .network_layer import DenseNetworkLayer
from .optimizers import AbstractOptimizer, get_optimizer


class PerceptronNetwork:
    """Multilayer perceptron: a chain of dense layers with a loss and an optimizer."""

    def __init__(self, network_layers: list[DenseNetworkLayer], loss_name: str,
                 optimizer_name: str, *, seed: int | None = None,
                 loss_parameters: dict[str, Any] | None = None,
                 optimizer_parameters: dict[str, Any] | None = None) -> None:
        """Initializes the network.

        Args:
            network_layers (list[DenseNetworkLayer]): Layers in forward order.
            loss_name (str): Registered loss name.
            optimizer_name (str): Registered optimizer name.
            seed (int | None): Seed for shuffling the training data.
            loss_parameters (dict[str, Any] | None): Extra keyword arguments for the loss.
            optimizer_parameters (dict[str, Any] | None): Extra keyword arguments for the optimizer.

        Raises:
            ValueError: If there are no layers or adjacent layer sizes differ.
            KeyError: If the loss or optimizer name is not registered.
        """
        if not network_layers:
            raise ValueError("network_layers must not be empty")
        for index, (previous, current) in enumerate(pairwise(network_layers)):
            out_features = previous.config["out_features"]
            in_features = current.config["in_features"]
            if out_features != in_features:
                raise ValueError(f"Layer {index} outputs {out_features} features, "
                                 f"but layer {index + 1} expects {in_features}")
        self._network_layers: list[DenseNetworkLayer] = list(network_layers)
        self._loss_name: str = loss_name
        self._optimizer_name: str = optimizer_name
        self._seed: int | None = seed
        self._loss_parameters: dict[str, Any] = dict(loss_parameters or {})
        self._optimizer_parameters: dict[str, Any] = dict(optimizer_parameters or {})

        self._loss_function: AbstractLossFunction = get_loss_function(loss_name, **self._loss_parameters)
        self._optimizer: AbstractOptimizer = get_optimizer(optimizer_name, **self._optimizer_parameters)
        self._rng = np.random.default_rng(seed)

    def predict(self, input_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the network output.

        Args:
            input_matrix (NDArray[np.float64]): Input of shape (batch, in_features).

        Returns:
            NDArray[np.float64]: Output of the last layer, shape (batch, out_features).
        """
        matrix: NDArray[np.float64] = np.asarray(input_matrix, dtype=np.float64)
        for layer in self._network_layers:
            matrix = layer.forward(matrix)
        return matrix

    def evaluate(self, input_matrix: NDArray[np.float64], target: NDArray[np.float64],
                 threshold: float = 0.5) -> tuple[float, float | None]:
        """Computes the loss and, for classification tasks, the accuracy.

        The task type is taken from the loss function. For binary
        classification an output is counted as the positive class when its
        probability exceeds threshold. For multiclass classification the
        predicted class is the column with the largest output, and target
        must be one-hot.

        Args:
            input_matrix (NDArray[np.float64]): Input of shape (batch, in_features).
            target (NDArray[np.float64]): Targets of shape (batch, out_features).
            threshold (float): Probability threshold for binary classification,
                in the open interval (0, 1).

        Returns:
            tuple[float, float | None]: The loss and the accuracy in [0, 1],
                or None for regression.

        Raises:
            ValueError: If threshold is not in (0, 1).
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in the open interval (0, 1), got {threshold!r}")
        target = np.asarray(target, dtype=np.float64)
        output_matrix: NDArray[np.float64] = self.predict(input_matrix)
        loss: float = self._loss_function.forward(output_matrix, target)

        match self._loss_function.task_type:
            case TaskType.REGRESSION:
                return loss, None
            case TaskType.BINARY_CLASSIFICATION:
                if self._loss_function.from_logits:
                    current_threshold: float = float(np.log(threshold / (1.0 - threshold)))
                else:
                    current_threshold = threshold
                correct = (output_matrix > current_threshold) == (target > threshold)
            case TaskType.MULTICLASS_CLASSIFICATION:
                correct = output_matrix.argmax(axis=1) == target.argmax(axis=1)
        return loss, float(np.mean(correct))

    def fit(self, input_matrix: NDArray[np.float64], target: NDArray[np.float64],
            *, epochs: int = 100, batch_size: int | None = None,
            validation: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None) -> dict[str, list[float]]:
        """Trains the network with mini-batch gradient descent.

        Args:
            input_matrix (NDArray[np.float64]): Input of shape (n_samples, in_features).
            target (NDArray[np.float64]): Targets of shape (n_samples, out_features).
            epochs (int): Number of passes over the training data.
            batch_size (int | None): Samples per update.
            validation (tuple[NDArray[np.float64], NDArray[np.float64]] | None):
                Optional (input, target) pair evaluated after every epoch.

        Returns:
            dict[str, list[float]]: Per-epoch "loss" and "acc", plus
                "val_loss" and "val_acc" when validation is given.

        Raises:
            ValueError: If epochs or batch_size is not a positive integer,
                or input and target have a different number of samples.
        """
        input_matrix = np.asarray(input_matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if len(target) != len(input_matrix):
            raise ValueError(f"input has {len(input_matrix)} samples, but target has {len(target)}")
        if epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}") 
        batch_size = batch_size if batch_size is not None else len(input_matrix)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        history: dict[str, list[float]] = {"loss": []}
        is_classification: bool = self._loss_function.task_type is not TaskType.REGRESSION

        if is_classification:
            history["acc"] = []
        if validation is not None:
            history["val_loss"] = []
            if is_classification:
                history["val_acc"] = []
    
        for epoch in track(range(epochs), description="Training..."):
            order: NDArray[np.int64] = self._rng.permutation(len(input_matrix))
            for start in range(0, len(input_matrix), batch_size):
                batch: NDArray[np.int64] = order[start : start + batch_size]
                self._train_step(input_matrix[batch], target[batch])
            loss, acc = self.evaluate(input_matrix, target)
            history["loss"].append(loss)
            message: str = f"[bold blue]Epoch {epoch + 1}/{epochs}[/bold blue] - loss: {loss:.4f}"

            if acc is not None:
                history["acc"].append(acc)
                message += f" - acc: {acc:.4f}"
            if validation is not None:
                val_loss, val_acc = self.evaluate(*validation)
                history["val_loss"].append(val_loss)
                message += f" - val_loss: {val_loss:.4f}"
                if val_acc is not None:
                    history["val_acc"].append(val_acc)
                    message += f" - val_acc: {val_acc:.4f}"
            
            rich_print(message)
        return history

    def _train_step(self, input_matrix: NDArray[np.float64], target: NDArray[np.float64]) -> None:
        """Runs forward and backward passes on one batch and updates parameters."""
        output_matrix: NDArray[np.float64] = self.predict(input_matrix)
        self._loss_function.forward(output_matrix, target)
        output_gradient: NDArray[np.float64] = self._loss_function.backward()
        for layer in reversed(self._network_layers):
            output_gradient = layer.backward(output_gradient)
        self._optimizer.update_parameters(self._network_layers)

    def save_network(self, path: str | Path, **extra: NDArray[Any]) -> Path:
        """Saves the configuration and parameters to an .npz file.

        Args:
            path (str | Path): Destination file. The ".npz" suffix is added if missing.
            **extra (NDArray[Any]): Extra arrays to store.

        Returns:
            Path: The path the file was written to.
        """
        path = Path(path).with_suffix(".npz")
        network_config: dict[str, Any] = {
            "layers": [layer.config for layer in self._network_layers],
            "loss_name": self._loss_name,
            "optimizer_name": self._optimizer_name,
            "seed": self._seed,
            "loss_parameters": self._loss_parameters,
            "optimizer_parameters": self._optimizer_parameters,
        }

        data: dict[str, NDArray[Any]] = {"config": np.array(json.dumps(network_config))}
        for index, layer in enumerate(self._network_layers):
            for name, parameter in layer.parameters.items():
                data[f"layer{index}_{name}"] = parameter
        for name, value in extra.items():
            data[f"extra_{name}"] = np.asarray(value)
        np.savez(path, **data)
        return path
 
    @classmethod
    def load_network(cls, path: str | Path) -> tuple[Self, dict[str, NDArray[Any]]]:
        """Loads a network saved by save_network.

        Args:
            path (str | Path): The saved file. The ".npz" suffix is added if missing.

        Returns:
            tuple[Self, dict[str, NDArray[Any]]]: The network and the extra
                arrays passed to save_network.
        """
        path = Path(path).with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"].item()))
            layers: list[DenseNetworkLayer] = []
            for index, layer_config in enumerate(config["layers"]):
                layer = DenseNetworkLayer(**layer_config)
                layer.set_parameters({name: data[f"layer{index}_{name}"] for name in layer.parameters})
                layers.append(layer)
            extra = {key.removeprefix("extra_"): data[key] for key in data.files if key.startswith("extra_")}
        network = cls(layers, config["loss_name"], config["optimizer_name"], seed=config["seed"],
                      loss_parameters=config["loss_parameters"], optimizer_parameters=config["optimizer_parameters"])
        return network, extra
