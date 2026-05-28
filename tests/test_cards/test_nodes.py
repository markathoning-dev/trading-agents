import pytest
from trading_agent.nodes.registry import NODE_REGISTRY, get_node, list_nodes
from trading_agent.nodes.pre_trade import MomentumAnalyzer, ReversionAnalyzer, VolatilityAnalyzer, HoldReinforcer
from trading_agent.nodes.post_trade import StopLossSentinel, PositionSizer, TrendFilter, PanicSell
from trading_agent.nodes.reward import VolatilityReward, DrawdownImmune


def create_test_state(**kwargs):
    state = {
        "cash": 10000.0,
        "shares": 0,
        "price": 100.0,
        "price_history": [100.0],
        "portfolio_values": [10000.0],
        "peak_value": 10000.0,
        "action": "",
        "trade_cost": 0.0,
        "reward": 0.0,
        "total_reward": 0.0,
        "step": 0,
        "done": False,
    }
    state.update(kwargs)
    return state


def test_node_registry():
    assert "momentum_analyzer" in NODE_REGISTRY
    assert "stop_loss_sentinel" in NODE_REGISTRY
    assert "volatility_reward" in NODE_REGISTRY


def test_get_node():
    node = get_node("momentum_analyzer")
    assert node is not None
    assert node.name == "momentum_analyzer"
    assert node.position == "pre_trade"


def test_list_nodes():
    nodes = list_nodes()
    assert len(nodes) >= 8
    assert any(n["name"] == "momentum_analyzer" for n in nodes)


def test_momentum_analyzer_short_history():
    state = create_test_state(price_history=[100.0, 101.0, 102.0])
    node = MomentumAnalyzer()
    result = node(state)
    assert result["momentum_signal"] == 0.0


def test_momentum_analyzer_with_history():
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    state = create_test_state(price_history=prices)
    node = MomentumAnalyzer()
    result = node(state)
    assert result["momentum_signal"] != 0.0


def test_reversion_analyzer_short_history():
    state = create_test_state(price_history=[100.0] * 5)
    node = ReversionAnalyzer()
    result = node(state)
    assert result["reversion_signal"] == 0.0


def test_reversion_analyzer_with_history():
    prices = [100.0] * 19 + [110.0]
    state = create_test_state(price_history=prices)
    node = ReversionAnalyzer()
    result = node(state)
    assert result["reversion_signal"] < 0


def test_volatility_analyzer_short_history():
    state = create_test_state(price_history=[100.0] * 5)
    node = VolatilityAnalyzer()
    result = node(state)
    assert result["volatility_signal"] == 0.0


def test_volatility_analyzer_with_history():
    prices = [100.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0, 106.0]
    state = create_test_state(price_history=prices)
    node = VolatilityAnalyzer()
    result = node(state)
    assert result["volatility_signal"] > 0


def test_stop_loss_no_trigger():
    state = create_test_state(portfolio_values=[10000.0, 10100.0])
    node = StopLossSentinel()
    result = node(state)
    assert result["shares"] == state["shares"]


def test_stop_loss_trigger():
    state = create_test_state(
        cash=5000.0,
        shares=50,
        price=85.0,
        portfolio_values=[10000.0, 8500.0],
        peak_value=10000.0,
    )
    node = StopLossSentinel()
    result = node(state)
    assert result["shares"] == 0
    assert result["action"] == "STOP_LOSS"


def test_position_sizer_within_limit():
    state = create_test_state(cash=5000.0, shares=30, price=100.0)
    node = PositionSizer()
    result = node(state)
    assert result["shares"] == 30


def test_position_sizer_over_limit():
    state = create_test_state(cash=2000.0, shares=80, price=100.0)
    node = PositionSizer()
    result = node(state)
    assert result["shares"] < 80


def test_panic_sell_no_trigger():
    state = create_test_state(portfolio_values=[10000.0, 10100.0, 10200.0, 10300.0])
    node = PanicSell()
    result = node(state)
    assert result["shares"] == state["shares"]


def test_panic_sell_trigger():
    state = create_test_state(
        cash=5000.0,
        shares=50,
        price=90.0,
        portfolio_values=[10000.0, 9500.0, 9000.0, 8500.0],
    )
    node = PanicSell()
    result = node(state)
    assert result["shares"] == 0
    assert result["action"] == "PANIC_SELL"


def test_drawdown_immune():
    state = create_test_state(
        cash=5000.0,
        shares=50,
        price=90.0,
        portfolio_values=[10000.0, 9500.0],
        peak_value=10000.0,
        trade_cost=10.0,
    )
    node = DrawdownImmune()
    result = node(state)
    assert result["reward"] == -510.0
    assert result["total_reward"] == -510.0


def test_trend_filter_short_history():
    state = create_test_state(price_history=[100.0] * 10)
    node = TrendFilter()
    result = node(state)
    assert result["trend_allowed_buy"] is True
    assert result["trend_allowed_sell"] is True


def test_trend_filter_downtrend():
    prices = [100.0] * 19 + [90.0]
    state = create_test_state(price_history=prices)
    node = TrendFilter()
    result = node(state)
    assert result["trend_signal"] < 0
    assert result["trend_allowed_buy"] is False
    assert result["trend_allowed_sell"] is True


def test_trend_filter_uptrend():
    prices = [100.0] * 19 + [110.0]
    state = create_test_state(price_history=prices)
    node = TrendFilter()
    result = node(state)
    assert result["trend_signal"] > 0
    assert result["trend_allowed_buy"] is True
    assert result["trend_allowed_sell"] is False
