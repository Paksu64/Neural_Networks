import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
import numpy as np

class FlowerModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01

        self.petal_weight_blue = nn.Parameter(torch.tensor(-2.5))
        self.petal_weight_orange = nn.Parameter(torch.tensor(-1.5))
        self.sepal_weight_blue = nn.Parameter(torch.tensor(0.6))
        self.sepal_weight_orange = nn.Parameter(torch.tensor(0.4))

        self.blue_bias = nn.Parameter(torch.tensor(1.6))
        self.orange_bias = nn.Parameter(torch.tensor(0.7))

        self.blue_setosa_weight = nn.Parameter(torch.tensor(-0.1))
        self.blue_versicolor_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.blue_virginica_weight = nn.Parameter(torch.tensor(-2.2))

        self.orange_setosa_weight = nn.Parameter(torch.tensor(1.5))
        self.orange_versicolor_weight = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        self.orange_virginica_weight = nn.Parameter(torch.tensor(3.7))

        self.setosa_bias = nn.Parameter(torch.tensor(0.0))
        self.versicolor_bias = nn.Parameter(torch.tensor(0.0))
        self.virginica_bias = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, petal_width, sepal_width):
        blue_path_sum = petal_width * self.petal_weight_blue + sepal_width * self.sepal_weight_blue + self.blue_bias
        orange_path_sum = petal_width * self.petal_weight_orange + sepal_width * self.sepal_weight_orange + self.orange_bias
        blue_relu = F.relu(blue_path_sum)
        orange_relu = F.relu(orange_path_sum)
        setosa_logit = blue_relu * self.blue_setosa_weight + orange_relu * self.orange_setosa_weight + self.setosa_bias
        versicolor_logit = blue_relu * self.blue_versicolor_weight + orange_relu * self.orange_versicolor_weight + self.versicolor_bias
        virginica_logit = blue_relu * self.blue_virginica_weight + orange_relu * self.orange_virginica_weight + self.virginica_bias
        logits = torch.stack([setosa_logit, versicolor_logit, virginica_logit], dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        petal_width, sepal_width, species = batch
        logits = self.forward(petal_width, sepal_width)
        probabilities = F.softmax(logits, dim=1)
        correct_probs = probabilities[torch.arange(len(species)), species]
        cross_entropy = -torch.log(correct_probs)
        loss = cross_entropy.sum()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

petal_widths = torch.tensor([0.04, 1.0, 0.5])
sepal_widths = torch.tensor([0.42, 0.54, 0.37])
species = torch.tensor([0, 2, 1])
dataset = TensorDataset(petal_widths, sepal_widths, species)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = FlowerModel()
trainer = L.Trainer(max_epochs=50)
tuner = Tuner(trainer)
lr_find_results = tuner.lr_find(model, train_dataloaders=dataloader, min_lr=0.001, max_lr=0.1, early_stop_threshold=None)
new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr

trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=dataloader)

test_petal = torch.tensor([0.5, 0.8])
test_sepal = torch.tensor([0.4, 0.6])
logits = model(test_petal, test_sepal)
probs = F.softmax(logits, dim=1)
predicted_indices = torch.argmax(probs, dim=1)
print("Probabilities:\n", probs)
print("Predicted class indices:", predicted_indices)

pw = np.linspace(0, 1, 100)
sw = np.linspace(0, 1, 100)
P, S = np.meshgrid(pw, sw)
grid_petal = torch.tensor(P, dtype=torch.float32).reshape(-1)
grid_sepal = torch.tensor(S, dtype=torch.float32).reshape(-1)
logits_grid = model(grid_petal, grid_sepal)
probs_grid = F.softmax(logits_grid, dim=1).detach().numpy()
probs_setosa = probs_grid[:, 0].reshape(P.shape)
probs_versicolor = probs_grid[:, 1].reshape(P.shape)
probs_virginica = probs_grid[:, 2].reshape(P.shape)

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

surf1 = ax1.plot_surface(P, S, probs_setosa, cmap="viridis")
ax1.set_title("Setosa")
ax1.set_xlabel("Petal Width")
ax1.set_ylabel("Sepal Width")
ax1.set_zlabel("Probability")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

surf2 = ax2.plot_surface(P, S, probs_versicolor, cmap="viridis")
ax2.set_title("Versicolor")
ax2.set_xlabel("Petal Width")
ax2.set_ylabel("Sepal Width")
ax2.set_zlabel("Probability")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

surf3 = ax3.plot_surface(P, S, probs_virginica, cmap="viridis")
ax3.set_title("Virginica")
ax3.set_xlabel("Petal Width")
ax3.set_ylabel("Sepal Width")
ax3.set_zlabel("Probability")
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
