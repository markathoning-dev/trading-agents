from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict


@dataclass
class Order:
    price: float
    quantity: int
    order_id: int
    timestamp: float
    side: str


@dataclass
class Trade:
    price: float
    quantity: int
    aggressor_side: str
    timestamp: float


@dataclass
class LOBState:
    mid_price: float
    spread: float
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_prices: np.ndarray
    ask_volumes: np.ndarray
    last_trades: list[Trade]


class OrderBook:
    def __init__(self, levels: int = 10):
        self.levels = levels
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self.next_id = 1
        self.trades: list[Trade] = []
        self.time = 0.0

    def add_limit_order(self, side: str, price: float, quantity: int) -> list[Trade]:
        trades: list[Trade] = []
        if side == "BUY":
            remaining = quantity
            while remaining > 0 and self.asks and self.asks[0].price <= price:
                best = self.asks[0]
                fill = min(remaining, best.quantity)
                trades.append(Trade(best.price, fill, "BUY", self.time))
                self.time += 1e-6
                remaining -= fill
                best.quantity -= fill
                if best.quantity <= 0:
                    self.asks.pop(0)
            if remaining > 0:
                self.bids.append(Order(price, remaining, self.next_id, self.time, "BUY"))
                self.next_id += 1
                self.bids.sort(key=lambda o: (-o.price, o.timestamp))
        else:
            remaining = quantity
            while remaining > 0 and self.bids and self.bids[0].price >= price:
                best = self.bids[0]
                fill = min(remaining, best.quantity)
                trades.append(Trade(best.price, fill, "SELL", self.time))
                self.time += 1e-6
                remaining -= fill
                best.quantity -= fill
                if best.quantity <= 0:
                    self.bids.pop(0)
            if remaining > 0:
                self.asks.append(Order(price, remaining, self.next_id, self.time, "SELL"))
                self.next_id += 1
                self.asks.sort(key=lambda o: (o.price, o.timestamp))
        self.trades.extend(trades)
        self.time += 1.0
        return trades

    def market_order(self, side: str, quantity: int) -> list[Trade]:
        trades: list[Trade] = []
        if side == "BUY":
            remaining = quantity
            while remaining > 0 and self.asks:
                best = self.asks[0]
                fill = min(remaining, best.quantity)
                trades.append(Trade(best.price, fill, "BUY", self.time))
                self.time += 1e-6
                remaining -= fill
                best.quantity -= fill
                if best.quantity <= 0:
                    self.asks.pop(0)
            if remaining > 0:
                pass
        else:
            remaining = quantity
            while remaining > 0 and self.bids:
                best = self.bids[0]
                fill = min(remaining, best.quantity)
                trades.append(Trade(best.price, fill, "SELL", self.time))
                self.time += 1e-6
                remaining -= fill
                best.quantity -= fill
                if best.quantity <= 0:
                    self.bids.pop(0)
            if remaining > 0:
                pass
        self.trades.extend(trades)
        self.time += 1.0
        return trades

    def cancel_order(self, order_id: int) -> bool:
        for book in (self.bids, self.asks):
            for i, o in enumerate(book):
                if o.order_id == order_id:
                    book.pop(i)
                    return True
        return False

    def get_lob_state(self) -> LOBState:
        best_bid = self.bids[0].price if self.bids else 0.0
        best_ask = self.asks[0].price if self.asks else float("inf")

        if self.bids and self.asks and best_bid > 0 and best_ask < float("inf"):
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
        else:
            mid = 100.0
            spread = 0.01

        bid_vol = sum(o.quantity for o in self.bids) if self.bids else 0
        ask_vol = sum(o.quantity for o in self.asks) if self.asks else 0

        bid_levels = sorted(set(o.price for o in self.bids), reverse=True)[:self.levels]
        ask_levels = sorted(set(o.price for o in self.asks))[:self.levels]

        bid_prices = np.zeros(self.levels, dtype=np.float32)
        bid_volumes = np.zeros(self.levels, dtype=np.float32)
        ask_prices = np.zeros(self.levels, dtype=np.float32)
        ask_volumes = np.zeros(self.levels, dtype=np.float32)

        for i, px in enumerate(bid_levels):
            bid_prices[i] = px
            bid_volumes[i] = sum(o.quantity for o in self.bids if o.price == px)

        for i, px in enumerate(ask_levels):
            ask_prices[i] = px
            ask_volumes[i] = sum(o.quantity for o in self.asks if o.price == px)

        recent_trades = self.trades[-10:] if self.trades else []

        return LOBState(
            mid_price=float(mid),
            spread=float(spread),
            best_bid=float(best_bid) if best_bid > 0 else 99.0,
            best_ask=float(best_ask) if best_ask < float("inf") else 101.0,
            bid_volume=float(bid_vol),
            ask_volume=float(ask_vol),
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            last_trades=recent_trades,
        )

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.trades.clear()
        self.next_id = 1
        self.time = 0.0


class LOBExchange:
    def __init__(self, feature_extractor: Callable | None = None, levels: int = 10):
        self.book = OrderBook(levels=levels)
        self.feature_extractor = feature_extractor
        self._trades_since_last: list[Trade] = []

    def process_action(
        self,
        action_type: int,
        side: int,
        price_offset: float,
        quantity: float,
        mid_price: float,
    ) -> list[Trade]:
        side_str = "BUY" if side == 0 else "SELL"
        qty = max(1, int(quantity * 100))
        price = mid_price * (1 + price_offset)

        if action_type == 0:
            trades = self.book.market_order(side_str, qty)
        elif action_type == 1:
            trades = self.book.add_limit_order(side_str, max(0.01, price), qty)
        elif action_type == 2:
            trades = self.book.add_limit_order(side_str, max(0.01, price), qty)
        elif action_type == 3:
            trades = []

        self._trades_since_last = trades
        return trades

    def get_lob_state(self) -> LOBState:
        return self.book.get_lob_state()

    def get_features(self) -> np.ndarray:
        if self.feature_extractor is None:
            from market_cgan.data.features import MarketFeatureExtractor
            return np.zeros(MarketFeatureExtractor.N_FEATURES, dtype=np.float32)
        state = self.get_lob_state()
        from market_cgan.data.lobster import LOBSnapshot
        snap = LOBSnapshot(
            timestamp=self.book.time,
            bid_prices=state.bid_prices,
            bid_volumes=state.bid_volumes,
            ask_prices=state.ask_prices,
            ask_volumes=state.ask_volumes,
        )
        return self.feature_extractor(snap)

    def reset(self):
        self.book.reset()
        self._trades_since_last = []
