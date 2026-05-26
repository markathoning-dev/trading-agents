from typing import TypedDict, List, NotRequired


class AgentState(TypedDict):
    cash: float
    shares: int
    price: float
    price_history: List[float]
    portfolio_values: List[float]
    peak_value: float
    action: str
    trade_cost: float
    reward: float
    total_reward: float
    step: int
    done: bool
    lob_bid: NotRequired[float]
    lob_ask: NotRequired[float]
    lob_spread: NotRequired[float]
    lob_mid: NotRequired[float]
    fill_price: NotRequired[float]
    order_filled: NotRequired[bool]
