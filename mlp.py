import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_input_tensors() -> tuple:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    x_tensor = torch.tensor(x, dtype=torch.float32)

    y = np.array([[0], [1], [1], [0]])
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor

def implement_xor() -> nn.Module:
    X, Y = get_input_tensors()

    input_dim = 2
    hidden_dim = 2
    output_dim = 1

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    epochs = 2000
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()

    return model


