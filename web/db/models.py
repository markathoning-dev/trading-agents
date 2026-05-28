from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from web.db.database import Base


class StrategyCard(Base):
    __tablename__ = "strategy_cards"
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    rarity = Column(String(20), nullable=False)
    mana_cost = Column(Integer, nullable=False)
    description = Column(String(500))
    flavor_text = Column(String(500))
    stats = Column(JSON, default={})
    reward_type = Column(String(50))
    nodes = Column(JSON, default=[])
    prompt_modifier = Column(String(1000))


class Deck(Base):
    __tablename__ = "decks"
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    card_ids = Column(JSON, default=[])
    mana_budget = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    runs = relationship("BacktestRun", back_populates="deck")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    data_source = Column(String(100), default="random")
    config = Column(JSON, default={})
    status = Column(String(20), default="pending")
    deck_id = Column(String(50), ForeignKey("decks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    result = relationship("BacktestResult", uselist=False, back_populates="run")
    deck = relationship("Deck", back_populates="runs")

class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    final_portfolio_value = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    cumulative_reward = Column(Float)
    num_steps = Column(Integer)
    final_cash = Column(Float)
    final_shares = Column(Integer)
    run = relationship("BacktestRun", back_populates="result")

class BacktestStep(Base):
    __tablename__ = "backtest_steps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    step = Column(Integer)
    price = Column(Float)
    cash = Column(Float)
    shares = Column(Integer)
    action = Column(String(20))
    portfolio_value = Column(Float)
    reward = Column(Float)

class PinnModel(Base):
    __tablename__ = "pinn_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    pde_type = Column(String(50), default="black_scholes")
    architecture = Column(JSON, default={})
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

class PinnTraining(Base):
    __tablename__ = "pinn_training"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("pinn_models.id"), nullable=False)
    epochs = Column(Integer)
    final_loss = Column(Float)
    loss_history = Column(JSON, default=[])
