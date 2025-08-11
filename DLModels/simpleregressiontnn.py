import torch.nn as nn

from tropicallayer import TropicalLayer

# Define a simple regression neural network using a tropical layer at the output
class SimpleRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionTNN, self).__init__()

        # Define the architecture as a sequence of layers:
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),         # First linear layer: input_dim → hidden_dim
            nn.ReLU(),                                # ReLU activation
            nn.Linear(hidden_dim, hidden_dim // 2),   # Second linear layer: hidden_dim → hidden_dim // 2
            nn.ReLU(),                                # ReLU activation
            TropicalLayer(hidden_dim // 2, 1)         # Tropical layer instead of standard linear output
        )

    # Forward pass through the network
    def forward(self, x):
        return self.net(x)                            # Pass input through the defined sequence of layers
