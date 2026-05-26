import torch
import torch.nn as nn


class BarDiscriminator(nn.Module):
    def __init__(self, bar_dim: int = 6, feature_dim: int = 6):
        super().__init__()
        self.bar_dim = bar_dim
        self.feature_dim = feature_dim
        input_dim = bar_dim + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, bar: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bar, features], dim=-1)
        return torch.sigmoid(self.net(x))

    def forward_logits(self, bar: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bar, features], dim=-1)
        return self.net(x)