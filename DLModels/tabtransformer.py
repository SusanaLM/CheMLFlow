import torch.nn as nn

# Define a Transformer-based model for tabular data regression
class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()

        # Linear embedding layer: transforms each scalar input feature into a vector of size embed_dim
        self.token_emb = nn.Linear(1, embed_dim)  # Input shape: [batch, input_dim, 1] → [batch, input_dim, embed_dim]

        # Define a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,               # Embedding dimension (must match token_emb output)
            nhead=n_heads,                   # Number of attention heads
            dropout=dropout,                 # Dropout within the transformer layer
            dim_feedforward=embed_dim * 4,   # Size of the feedforward sublayer (usually 4x d_model)
            batch_first=True                 # Use shape [batch, seq, feature] instead of [seq, batch, feature]
        )

        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers              # Number of stacked transformer blocks
        )

        # Prediction head: maps transformer output to a single regression value
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),  # First layer: reduce dimension
            nn.ReLU(),                             # Activation
            nn.Dropout(dropout),                   # Dropout for regularization
            nn.Linear(embed_dim // 2, 1)           # Final layer: output a single value
        )


        # self.head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, embed_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim//2, 1)
        # )
