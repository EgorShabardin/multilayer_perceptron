import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multilayer_perceptron import DenseNetworkLayer, PerceptronNetwork

TRAIN_PATH = "data/tumor_class/train.csv"
VALIDATION_PATH = "data/tumor_class/validation.csv"
MODEL_PATH = "models/tumor_class.npz"
 
EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SEED = 42
 
train_data = pd.read_csv(TRAIN_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)
 
x_train = train_data.drop(columns=["id", "diagnosis"]).to_numpy(dtype=float)
x_validation = validation_data.drop(columns=["id", "diagnosis"]).to_numpy(dtype=float)
y_train = (train_data["diagnosis"] == "M").astype(int).to_numpy()
y_validation = (validation_data["diagnosis"] == "M").astype(int).to_numpy()
y_train, y_validation = np.eye(2)[y_train], np.eye(2)[y_validation]
 
mean, std = x_train.mean(axis=0), x_train.std(axis=0)
x_train = (x_train - mean) / std
x_validation = (x_validation - mean) / std
 
print("x_train shape :", x_train.shape)
print("x_valid shape :", x_validation.shape)

rng = np.random.default_rng(SEED)
layers = [
    DenseNetworkLayer(30, 8, "leaky_relu", "kaiming_normal", rng=rng),
    DenseNetworkLayer(8, 2, "identity", "kaiming_normal", rng=rng)
]
 
network = PerceptronNetwork(layers, "cce_with_logits", "sgd", seed=SEED, optimizer_parameters={"learning_rate": LEARNING_RATE})
history = network.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation=(x_validation, y_validation))
network.save_network(MODEL_PATH, mean=mean, std=std)

epochs = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
 
ax[0].plot(epochs, history["loss"], "r-", label="training_loss")
ax[0].plot(epochs, history["val_loss"], "b--", label="validation loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[0].grid()
 
ax[1].plot(epochs, history["acc"], label="training acc")
ax[1].plot(epochs, history["val_acc"], label="validation acc")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()
ax[1].grid()
 
fig.suptitle("Learning Curves")
plt.show()
