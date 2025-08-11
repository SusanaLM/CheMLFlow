import torch.nn as nn

from tropicallayer import TropicalLayer

# Define a deep neural network for regression using tropical geometry in the final layer
class DeepRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionTNN, self).__init__()

        layers = []              # List to store the layers of the network
        prev_dim = input_dim     # Initial input dimension

        # Build hidden layers
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),      # Linear layer: prev_dim → h_dim
                nn.BatchNorm1d(h_dim),           # Batch normalization to stabilize training
                nn.ReLU(),                       # ReLU activation function
                nn.Dropout(dropout_rate)         # Dropout for regularization
            ]
            prev_dim = h_dim                     # Update prev_dim for the next layer

        # Final output layer using a TropicalLayer (instead of a standard linear layer)
        layers.append(TropicalLayer(prev_dim, 1))

        # Combine all layers into a single sequential model
        self.net = nn.Sequential(*layers)

    # Forward pass through the network
    def forward(self, x):
        return self.net(x)                       # Pass input x through the network
