from trading_agent.core.reward import multicomponent_reward, aggressive_reward

def test_profit_only():
    r = multicomponent_reward(100, 110, 110, 0, risk_penalty_lambda=0.1)
    assert r == 10.0

def test_drawdown_penalty():
    r = multicomponent_reward(100, 90, 110, 0, risk_penalty_lambda=0.1)
    assert r == -12.0

def test_transaction_cost():
    r = multicomponent_reward(100, 110, 110, 5, risk_penalty_lambda=0.1)
    assert r == 5.0

def test_no_risk_penalty_when_above_peak():
    r = multicomponent_reward(100, 120, 110, 0, risk_penalty_lambda=0.1)
    assert r == 20.0

def test_zero_lambda_no_penalty():
    r = multicomponent_reward(100, 90, 110, 0, risk_penalty_lambda=0)
    assert r == -10.0

def test_negative_profit_with_drawdown():
    r = multicomponent_reward(100, 50, 110, 2, risk_penalty_lambda=0.5)
    assert r == -50 - 30 - 2


def test_aggressive_reward_profit():
    r = aggressive_reward(100, 110, 110, 0, volatility_lambda=0.2)
    assert r == 10 + 0.2 * 10


def test_aggressive_reward_volatility_bonus():
    r_profit = aggressive_reward(100, 110, 110, 0, 0.2)
    r_loss = aggressive_reward(100, 90, 110, 0, 0.2)
    assert r_profit > 10
    assert r_loss > -10


def test_aggressive_reward_no_drawdown_penalty():
    r_with_dd = aggressive_reward(100, 80, 120, 0, 0.2)
    expected = -20 + 0.2 * 20
    assert r_with_dd == expected


def test_aggressive_reward_with_trade_cost():
    r = aggressive_reward(100, 110, 110, 5, 0.2)
    assert r == 10 + 0.2 * 10 - 5
