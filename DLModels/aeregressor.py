import torch.nn as nn

# Define a regression model that uses a pretrained autoencoder encoder as a feature extractor
class AERegressor(nn.Module):
    def __init__(self, pretrained_encoder, bottleneck=64, dropout=0.1):
        super().__init__()

        self.encoder = pretrained_encoder  # Use an existing (possibly pretrained) encoder

        # Regression head: maps encoded features (bottleneck) to a single output
        self.head = nn.Sequential(
            nn.Linear(bottleneck, 128),     # Bottleneck → hidden layer
            nn.ReLU(),                      # ReLU activation
            nn.Dropout(dropout),            # Dropout for regularization
            nn.Linear(128, 1)               # Output layer: hidden → scalar prediction
        )

    # Forward method for prediction
    def forward(self, x):
        # Optionally freeze encoder by wrapping it in torch.no_grad()
        # with torch.no_grad():
        z = self.encoder(x)                 # Extract latent features from encoder
        return self.head(z).squeeze(1)     # Predict and squeeze output to shape [batch]
