import torch
import torch.nn as nn
import torch.nn.functional as F


class BarGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 64,
        feature_dim: int = 6,
        ref_price: float = 100.0,
        max_move: float = 0.05,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.ref_price = ref_price
        self.max_move = max_move
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

        self.open_head = nn.Linear(128, 1)
        self.close_head = nn.Linear(128, 1)
        self.high_head = nn.Linear(128, 1)
        self.low_head = nn.Linear(128, 1)
        self.volume_head = nn.Linear(128, 1)
        self.vwap_head = nn.Linear(128, 1)

    def forward(self, noise: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, features], dim=-1)
        h = self.net(x)

        o_raw = torch.tanh(self.open_head(h))
        c_raw = torch.tanh(self.close_head(h))
        v_raw = F.softplus(self.volume_head(h))
        vw_raw = torch.tanh(self.vwap_head(h))
        h_raw = F.softplus(self.high_head(h))
        l_raw = F.softplus(self.low_head(h))

        feature_ref = features[:, 3:4] * self.ref_price
        ref = feature_ref.clamp(min=1.0)

        o = ref * (1.0 + o_raw * self.max_move)
        c = ref * (1.0 + c_raw * self.max_move)
        o_max = torch.max(o, c)
        o_min = torch.min(o, c)
        h = o_max + h_raw * ref * self.max_move
        l = o_min - l_raw * ref * self.max_move
        v = v_raw * 10000
        vw = ref * (1.0 + vw_raw * self.max_move)

        return torch.cat([o, h, l, c, v, vw], dim=1)