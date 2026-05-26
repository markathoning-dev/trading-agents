from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from web.db.database import Base

class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    data_source = Column(String(100), default="random")
    config = Column(JSON, default={})
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    result = relationship("BacktestResult", uselist=False, back_populates="run")

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
