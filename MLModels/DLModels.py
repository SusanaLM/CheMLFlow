import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

class TropicalLayer( nn.Module ):
    def __init__( self, in_features, out_features ):
        super( TropicalLayer, self ).__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter( torch.randn( out_features, in_features ))
        nn.init.normal_( self. weight )

    def forward ( self, x):
        """ Returns negative tropical distance between x and self . weight . """
        result_addition = x.unsqueeze(1) - self.weight # [B, 1 , in] - [out , in] -> [B, out , in]
        return torch.min( result_addition, dim = -1).values - torch.max( result_addition, dim = -1).values # [B, out , in] -> [B, out ]


class SimpleRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class SimpleRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressionTNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            TropicalLayer(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class DeepRegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepRegressionTNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(DeepRegressionTNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            prev_dim = h_dim
        layers.append(TropicalLayer(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    
class GRURegressor(nn.Module):
    def __init__(self,
                 seq_len,
                 input_size=1,
                 hidden_size=512,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.2):
        super(GRURegressor, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)
        factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is shape [batch, seq_len]; treat each feature as one step
        x = x.unsqueeze(-1)            # → [batch, seq_len, 1]
        gru_out, _ = self.gru(x)       # → [batch, seq_len, hidden_size * factor]
        last = gru_out[:, -1, :]       # pick the last time step
        return self.head(last).squeeze(1)

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            out = block(x)
            x = x + out        # residual connection
        return self.head(x).squeeze(1)

class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Linear(1, embed_dim)  # each scalar → vector
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout, dim_feedforward=embed_dim*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1)
        )

    def forward(self, x):
        # x: [batch, D] → [batch, D, 1] → embed to [batch, D, E]
        emb = self.token_emb(x.unsqueeze(-1))
        # transformer expects [seq_len, batch, embed]
        t = emb.permute(1, 0, 2)
        out = self.transformer(t)
        # out: [D, batch, E] → pool across D
        pooled = out.mean(dim=0)
        return self.head(pooled).squeeze(1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AERegressor(nn.Module):
    def __init__(self, pretrained_encoder, bottleneck=32, dropout=0.2):
        super().__init__()
        self.encoder = pretrained_encoder
        self.head = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        with torch.no_grad():   # optionally freeze encoder
            z = self.encoder(x)
        return self.head(z).squeeze(1)



