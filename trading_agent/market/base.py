from abc import ABC, abstractmethod

class MarketDataSource(ABC):
    @abstractmethod
    def step(self) -> float:
        ...

    def reset(self) -> None:
        ...
