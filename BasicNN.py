import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as pltei
import seaborn as sns

'''class BasicNN(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.70), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

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
'''

class BasicNN_Train(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.70), requires_grad=False)

        # requires_gradient = True because value needs gradient descent to be optimized
        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

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
    
inputs = torch.tensor([0.0, 0.5, 1.0])
labels = torch.tensor([0.0, 1.0, 0.0])

model = BasicNN_Train()
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    
    total_loss = 0

    for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]
        output = model(input_i)

        loss = (label_i - output)**2

        loss.backward()

        total_loss += float(loss)
    
    if total_loss < 0.0001:
        break

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}:")
    print(f"  Total Loss = {total_loss}")
    print(f"  Final Bias = {model.final_bias.data}")
    print("-" * 50)