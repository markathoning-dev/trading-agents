import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 64, feature_dim: int = 42):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        input_dim = noise_dim + feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
        )

        self.action_type_head = nn.Linear(128, 4)
        self.side_head = nn.Linear(128, 2)
        self.price_head = nn.Linear(128, 1)
        self.quantity_head = nn.Linear(128, 1)

    def forward(self, noise: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([noise, features], dim=-1)
        h = self.net(x)

        action_type_logits = self.action_type_head(h)
        side_logits = self.side_head(h)
        price_offset = torch.tanh(self.price_head(h))
        quantity = torch.sigmoid(self.quantity_head(h))

        return {
            "action_type": F.softmax(action_type_logits, dim=-1),
            "action_type_logits": action_type_logits,
            "side": F.softmax(side_logits, dim=-1),
            "side_logits": side_logits,
            "price_offset": price_offset,
            "quantity": quantity,
        }

    def sample(self, noise: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(noise, features)
        action_type = torch.multinomial(out["action_type"], 1).squeeze(-1)
        side = torch.multinomial(out["side"], 1).squeeze(-1)
        return {
            "action_type": action_type,
            "side": side,
            "price_offset": out["price_offset"].squeeze(-1),
            "quantity": out["quantity"].squeeze(-1),
        }
