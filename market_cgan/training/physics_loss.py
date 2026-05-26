import torch

DEFAULT_ACTION_DIST = (0.3, 0.3, 0.3, 0.1)
DEFAULT_MEAN_QUANTITY = 0.15


def spread_respecting_price_loss(
    price_offset: torch.Tensor,
    action_type_probs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    mid_price = features[:, 0:1] * 100.0
    spread = mid_price * features[:, 1:2]
    limit_mask = (action_type_probs[:, 1:2] + action_type_probs[:, 2:3]).detach()
    violation = torch.relu(torch.abs(price_offset) * mid_price - spread / 2)
    return (violation * limit_mask).mean()


def action_distribution_loss(
    action_type_probs: torch.Tensor,
    target: tuple[float, ...] = DEFAULT_ACTION_DIST,
) -> torch.Tensor:
    target_t = torch.tensor(target, device=action_type_probs.device, dtype=torch.float32)
    empirical = action_type_probs.mean(dim=0)
    return torch.nn.functional.mse_loss(empirical, target_t)


def side_balance_loss(
    side_probs: torch.Tensor,
) -> torch.Tensor:
    buy_prob = side_probs[:, 0].mean()
    return (buy_prob - 0.5).pow(2)


def quantity_matching_loss(
    quantity: torch.Tensor,
    target: float = DEFAULT_MEAN_QUANTITY,
) -> torch.Tensor:
    return (quantity.mean() - target).pow(2)


def physics_informed_loss(
    fake_outputs: dict[str, torch.Tensor],
    features: torch.Tensor,
    action_dist_target: tuple[float, ...] = DEFAULT_ACTION_DIST,
    mean_quantity_target: float = DEFAULT_MEAN_QUANTITY,
) -> dict[str, torch.Tensor]:
    return {
        "physics_spread": spread_respecting_price_loss(
            fake_outputs["price_offset"], fake_outputs["action_type"], features
        ),
        "physics_action_dist": action_distribution_loss(
            fake_outputs["action_type"], action_dist_target
        ),
        "physics_side_balance": side_balance_loss(fake_outputs["side"]),
        "physics_quantity": quantity_matching_loss(fake_outputs["quantity"], mean_quantity_target),
    }
