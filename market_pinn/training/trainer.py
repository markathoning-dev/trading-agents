import torch
from torch.utils.data import DataLoader
from market_pinn.training.dataset import MarketDataset
from market_pinn.training.losses import pinn_loss


def train_pinn(model, dataset, epochs=100, lr=1e-3, r=0.05, sigma=0.2, w_data=1.0, w_pde=0.1, log_interval=10):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for tb, sb, targetb in loader:
            tb.requires_grad_(True)
            sb.requires_grad_(True)
            optimizer.zero_grad()
            loss = pinn_loss(model, tb, sb, targetb, r, sigma, w_data, w_pde)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")
    return history
