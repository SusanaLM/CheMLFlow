import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from gruregressor import GRURegressor  # Custom GRU-based regressor for sequence data

# Define a custom deep learning regressor compatible with scikit-learn's API
class DLRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 model_class,              # Class of the PyTorch model to use
                 model_kwargs=None,        # Keyword arguments for the model constructor
                 epochs=100,               # Number of training epochs
                 batch_size=32,            # Size of each mini-batch
                 learning_rate=1e-3,       # Learning rate for optimizer
                 verbose=False):           # Whether to print training logs
        # Save initialization parameters
        self.model_class   = model_class
        self.model_kwargs  = model_kwargs or {}
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.verbose       = verbose

        # Standardize inputs using sklearn's StandardScaler
        self.scaler        = StandardScaler()

        # Use GPU if available
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholder for the actual model instance
        self.model         = None

    def fit(self, X, y):
        # Scale the input features
        Xs = self.scaler.fit_transform(X)

        # Ensure y is shaped as a column vector
        Ys = np.array(y).reshape(-1, 1)

        # Check if the model is a sequence model (e.g. GRU-based)
        is_sequence = issubclass(self.model_class, GRURegressor)

        # Convert inputs to PyTorch tensors
        Xt = torch.tensor(Xs, dtype=torch.float32)
        if is_sequence:
            Xt = Xt.unsqueeze(2)  # Add a third dimension: [N, D] → [N, D, 1]
        Yt = torch.tensor(Ys, dtype=torch.float32).to(self.device)

        # Create a PyTorch dataset and data loader for mini-batch training
        ds     = TensorDataset(Xt.to(self.device), Yt)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Instantiate the model and move it to the correct device
        self.model = self.model_class(**self.model_kwargs).to(self.device)

        # Flatten GRU parameters for more efficient training (if applicable)
        if is_sequence and hasattr(self.model, 'gru'):
            self.model.gru.flatten_parameters()

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()  # Mean squared error for regression

        # Training loop
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()                # Clear gradients
                preds = self.model(bx)               # Forward pass
                loss  = criterion(preds.unsqueeze(1), by)  # Compute loss
                loss.backward()                      # Backpropagation
                optimizer.step()                     # Update weights
                total_loss += loss.item()            # Track loss
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} — loss: {total_loss/len(loader):.4f}")
        return self

    def predict(self, X):
        # Apply the same scaling to inputs as during training
        Xs = self.scaler.transform(X)

        # Convert to tensor
        Xt = torch.tensor(Xs, dtype=torch.float32)
        if issubclass(self.model_class, GRURegressor):
            Xt = Xt.unsqueeze(2)  # Add sequence dimension if needed
        Xt = Xt.to(self.device)

        # Set model to evaluation mode and disable gradient tracking
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                batch = Xt[i:i+self.batch_size]
                p = self.model(batch)
                preds_list.append(p.cpu().numpy())  # Move predictions to CPU and store
        return np.concatenate(preds_list).flatten()  # Return as 1D array
