import torch.nn as nn

# Define a simple feedforward neural network for regression tasks
class SimpleRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionNN, self).__init__()
        
        # Define the network architecture
        # Input layer → Hidden layer (ReLU) → Hidden layer (ReLU) → Output layer
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),         # First linear layer: input_dim → hidden_dim
            nn.ReLU(),                                # ReLU activation
            nn.Linear(hidden_dim, hidden_dim // 2),   # Second linear layer: hidden_dim → hidden_dim // 2
            nn.ReLU(),                                # ReLU activation
            nn.Linear(hidden_dim // 2, 1)             # Final linear layer: output → single value
        )

    # Forward pass of the network
    def forward(self, x):
        return self.net(x)                            # Pass input through the sequential network
