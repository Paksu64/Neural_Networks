import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner import Tuner

import matplotlib.pyplot as plt
import seaborn as sns


class LightningNN_Train(L.LightningModule):

    def __init__(self):
        
        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(-9.0), requires_grad=True)

        self.final_bias = nn.Parameter(torch.tensor(15.0), requires_grad=True)
        self.learning_rate = 0.01

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu + scaled_bottom_relu + self.final_bias
        output = F.relu(input_to_final_relu)
        
        return output
    
    def configure_optimizers(self):

        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batchIndex):

        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = ((label_i - output_i)**2).sum()

        return loss

input_doses = torch.linspace(start=0, end=1, steps=11)
inputs = torch.tensor([0., 0.5, 1.] * 100)
labels = torch.tensor([0., 1., 0.] * 100)

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

model = LightningNN_Train()
trainer = L.Trainer(max_epochs = 50)
tuner = Tuner(trainer)

lr_find_results = tuner.lr_find(model,
                                train_dataloaders = dataloader,
                                min_lr = 0.001,
                                max_lr = 0.1,
                                early_stop_threshold = None)

new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr

print("Gradient for final_bias:", model.final_bias.grad)
print("Gradient for w11:", model.w11.grad)
print(model.w11.data)
print(model.final_bias.data)

trainer.fit(model, train_dataloaders = dataloader)
output_values = model(input_doses)

plt.plot(input_doses, output_values.detach(), marker='o')
plt.title('Dosage vs Efficiency')
plt.xlabel('Dosage')
plt.ylabel('Efficiency')
plt.axhline(y=0.0, color='g', linestyle='--')
plt.show()
