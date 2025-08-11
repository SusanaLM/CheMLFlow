import torch.nn as nn

# Define a deep feedforward neural network for regression with multiple hidden layers,
# batch normalization, ReLU activation, and dropout regularization
class DeepRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionNN, self).__init__()

        layers = []             # List to hold all the layers
        prev_dim = input_dim    # Start with the input dimension

        # Construct hidden layers
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),      # Linear layer: prev_dim → h_dim
                nn.BatchNorm1d(h_dim),           # Batch normalization for stable training
                nn.ReLU(),                       # ReLU activation
                nn.Dropout(dropout_rate)         # Dropout for regularization
            ]
            prev_dim = h_dim                     # Update previous dim for next layer

        # Final output layer: maps last hidden layer to a single output value
        layers.append(nn.Linear(prev_dim, 1))

        # Combine all layers into a sequential model
        self.net = nn.Sequential(*layers)

    # Forward pass through the network
    def forward(self, x):
        return self.net(x)                       # Pass input through the entire network
