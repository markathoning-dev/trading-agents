import numpy as np
from market_cgan.simulation.exchange import OrderBook, LOBExchange, Order


def test_order_book_market_buy():
    book = OrderBook()
    book.asks.append(Order(101.0, 100, 1, 0.0, "SELL"))
    book.bids.append(Order(99.0, 100, 2, 0.0, "BUY"))
    trades = book.market_order("BUY", 50)
    assert len(trades) == 1
    assert trades[0].price == 101.0
    assert trades[0].quantity == 50
    assert book.asks[0].quantity == 50


def test_order_book_market_sell():
    book = OrderBook()
    book.asks.append(Order(101.0, 100, 1, 0.0, "SELL"))
    book.bids.append(Order(99.0, 100, 2, 0.0, "BUY"))
    trades = book.market_order("SELL", 30)
    assert len(trades) == 1
    assert trades[0].price == 99.0
    assert trades[0].quantity == 30
    assert book.bids[0].quantity == 70


def test_order_book_limit_buy():
    book = OrderBook()
    book.asks.append(Order(101.0, 100, 1, 0.0, "SELL"))
    trades = book.add_limit_order("BUY", 100.5, 50)
    assert len(trades) == 0
    assert len(book.bids) == 1
    assert book.bids[0].price == 100.5
    assert book.bids[0].quantity == 50


def test_order_book_limit_buy_crossing():
    book = OrderBook()
    book.asks.append(Order(101.0, 100, 1, 0.0, "SELL"))
    trades = book.add_limit_order("BUY", 102.0, 150)
    assert len(trades) >= 1
    assert trades[0].price == 101.0


def test_order_book_cancel():
    book = OrderBook()
    book.bids.append(Order(100.0, 50, 1, 0.0, "BUY"))
    result = book.cancel_order(1)
    assert result
    assert len(book.bids) == 0


def test_order_book_cancel_nonexistent():
    book = OrderBook()
    result = book.cancel_order(999)
    assert not result


def test_exchange_process_market():
    exchange = LOBExchange()
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)
    trades = exchange.process_action(0, 0, 0.0, 0.5, 100.0)
    assert len(trades) >= 1


def test_exchange_get_lob_state():
    exchange = LOBExchange()
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)
    state = exchange.get_lob_state()
    assert state.best_bid == 99.0
    assert state.best_ask == 101.0
    assert state.spread == 2.0
    assert state.mid_price == 100.0


def test_exchange_reset():
    exchange = LOBExchange()
    exchange.book.add_limit_order("SELL", 101.0, 100)
    assert len(exchange.book.asks) == 1
    exchange.reset()
    assert len(exchange.book.asks) == 0
    assert len(exchange.book.bids) == 0


def test_exchange_get_features():
    exchange = LOBExchange()
    exchange.book.add_limit_order("SELL", 101.0, 100)
    exchange.book.add_limit_order("BUY", 99.0, 100)
    features = exchange.get_features()
    assert features.shape == (42,)
    assert np.all(np.isfinite(features))
