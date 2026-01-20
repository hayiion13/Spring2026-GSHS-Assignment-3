import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_input_tensors():
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

def implement_xor() -> nn.Module:
    set_seed(42)
    X, Y = get_input_tensors()

    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1)   # Sigmoid 제거
    )

    criterion = nn.BCEWithLogitsLoss()  # 변경
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

    # 결과 확인
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
        preds = (probs > 0.5).float()
        print("Predictions:")
        print(preds.squeeze().tolist())
        print("Targets:")
        print(Y.squeeze().tolist())

    return model
