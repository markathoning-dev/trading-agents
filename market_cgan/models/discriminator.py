import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, action_dim: int = 8, feature_dim: int = 42):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        input_dim = action_dim + feature_dim

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

    def forward(self, action: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([action, features], dim=-1)
        return torch.sigmoid(self.net(x))

    def forward_logits(self, action: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([action, features], dim=-1)
        return self.net(x)
