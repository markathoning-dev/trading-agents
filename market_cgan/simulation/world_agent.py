import torch
import numpy as np
from market_cgan.models.generator import Generator
from market_cgan.simulation.exchange import LOBExchange, LOBState


class WorldAgent:
    def __init__(
        self,
        generator: Generator,
        exchange: LOBExchange,
        noise_dim: int = 64,
        device: str | None = None,
    ):
        self.generator = generator
        self.exchange = exchange
        self.noise_dim = noise_dim
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.generator.to(self.device)
        self.generator.eval()

    def step(self) -> LOBState:
        features = self.exchange.get_features()
        state = self.exchange.get_lob_state()

        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        noise = torch.randn(1, self.noise_dim, device=self.device)

        with torch.no_grad():
            action = self.generator.sample(noise, feat_tensor)

        self.exchange.process_action(
            action_type=int(action["action_type"].item()),
            side=int(action["side"].item()),
            price_offset=float(action["price_offset"].item()),
            quantity=float(action["quantity"].item()),
            mid_price=state.mid_price,
        )

        return self.exchange.get_lob_state()

    def reset(self):
        self.exchange.reset()
