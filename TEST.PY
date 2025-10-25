import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import CSVLogger

import matplotlib.pyplot as plt
import seaborn as sns

# Use a Seaborn style for plotting
sns.set_theme(style="darkgrid")

class LightningNN_Train(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Top branch (fixed parameters)
        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        # Bottom branch (only w11 is trainable)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(-9.0), requires_grad=True)

        # Final bias is trainable
        self.final_bias = nn.Parameter(torch.tensor(15.0), requires_grad=True)
        self.learning_rate = 0.01

    def forward(self, x):
        # Compute the top branch
        top_input = x * self.w00 + self.b00
        top_output = F.relu(top_input)
        scaled_top = top_output * self.w01

        # Compute the bottom branch
        bottom_input = x * self.w10 + self.b10
        bottom_output = F.relu(bottom_input)
        scaled_bottom = bottom_output * self.w11

        # Combine both branches with final bias
        final_input = scaled_top + scaled_bottom + self.final_bias

        # Option 1: Use LeakyReLU to avoid dead zone (preferred for debugging)
        output = F.leaky_relu(final_input, negative_slope=0.01)
        # Option 2: Remove the final activation completely:
        # output = final_input

        return output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = ((y - y_hat) ** 2).mean()
        # Log the training loss so it appears in the CSV file
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

if __name__ == "__main__":
    # Create dataset: inputs [0, 0.5, 1] repeated and corresponding labels [0, 1, 0]
    inputs = torch.tensor([0., 0.5, 1.] * 100)
    labels = torch.tensor([0., 1., 0.] * 100)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set up CSVLogger so that Lightning logs training metrics to a CSV file
    csv_logger = CSVLogger(save_dir="logs", name="my_experiment")

    # Initialize the model
    model = LightningNN_Train()

    # Create the Trainer and attach the CSV logger
    trainer = L.Trainer(
        max_epochs=50,
        logger=csv_logger,
        enable_checkpointing=False
    )

    # Train the model
    trainer.fit(model, train_dataloaders=dataloader)

    # Evaluate model outputs for a range of input doses
    input_doses = torch.linspace(0, 1, 11)
    with torch.no_grad():
        output_values = model(input_doses)

    # Convert tensors to NumPy arrays for plotting
    doses = input_doses.numpy()
    efficiencies = output_values.numpy()

    # Create a line plot using Seaborn
    sns.lineplot(x=doses, y=efficiencies, marker="o")
    plt.title("Dosage vs Efficiency")
    plt.xlabel("Dosage")
    plt.ylabel("Efficiency")
    plt.axhline(y=0.0, color="g", linestyle="--")
    plt.show()

    # Note: The CSV logs will be saved in logs/my_experiment/version_0/metrics.csv
