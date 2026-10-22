import numpy as np
import pandas as pd

from multilayer_perceptron import PerceptronNetwork, get_loss_function

VALIDATION_PATH = "data/tumor_class/validation.csv"
MODEL_PATH = "models/tumor_class.npz"

data = pd.read_csv(VALIDATION_PATH)
net, extra = PerceptronNetwork.load_network(MODEL_PATH)
mean, std = extra["mean"], extra["std"]

x_data = data.iloc[:, 2:].to_numpy(dtype=np.float64)
y_data = (data.iloc[:, 1] == "M").astype(dtype=np.int64).to_numpy()
x_data = (x_data - mean) / std
y_data = np.eye(2)[y_data]

prediction = net.predict(x_data)
logit_malignant = prediction[:, 1:] - prediction[:, :1]
loss = get_loss_function("bce_with_logits").forward(logit_malignant, y_data[:, 1:])
accuracy = (prediction.argmax(axis=1) == y_data.argmax(axis=1)).mean()

print(f"binary cross-entropy : {loss:.4f}")
print(f"accuracy : {accuracy:.4f}")
