import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from web.db.database import Base
from web.db.models import BacktestRun, BacktestResult

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    yield sess
    sess.close()

def test_create_backtest_run(session):
    run = BacktestRun(model_name="gpt-4o-mini", status="running")
    session.add(run)
    session.commit()
    assert run.id is not None
    assert run.status == "running"

def test_create_result(session):
    run = BacktestRun(model_name="gpt-4o-mini", status="completed")
    session.add(run)
    session.flush()
    result = BacktestResult(run_id=run.id, total_return=0.05, sharpe_ratio=1.2)
    session.add(result)
    session.commit()
    assert result.id is not None
