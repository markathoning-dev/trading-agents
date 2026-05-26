from trading_agent.market.base import MarketDataSource

class YahooMarket(MarketDataSource):
    def __init__(self, symbol: str, period: str = "1y"):
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period)
        self.prices = hist["Close"].values
        self.idx = 0

    def step(self) -> float:
        if self.idx >= len(self.prices):
            raise StopIteration("End of data")
        price = self.prices[self.idx]
        self.idx += 1
        return float(price)

    def reset(self) -> None:
        self.idx = 0
