import numpy as np
from market_cgan.data.lobster import LOBSnapshot


class MarketFeatureExtractor:
    N_FEATURES = 42

    def __call__(self, snapshot: LOBSnapshot) -> np.ndarray:
        features = []

        bid_px = snapshot.bid_prices
        ask_px = snapshot.ask_prices
        bid_vol = snapshot.bid_volumes
        ask_vol = snapshot.ask_volumes

        best_bid = bid_px[0]
        best_ask = ask_px[0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        features.append(mid_price / 100.0)
        features.append(spread / (mid_price + 1e-8))
        features.append(best_bid / 100.0)
        features.append(best_ask / 100.0)
        features.append(bid_vol[0] / 100.0)
        features.append(ask_vol[0] / 100.0)

        bid_vol_total = bid_vol.sum() + 1e-8
        ask_vol_total = ask_vol.sum() + 1e-8
        imbalance = (bid_vol.sum() - ask_vol.sum()) / (bid_vol.sum() + ask_vol.sum() + 1e-8)
        features.append(imbalance)

        micro_price = (bid_vol[0] * best_ask + ask_vol[0] * best_bid) / (bid_vol[0] + ask_vol[0] + 1e-8)
        features.append(micro_price / 100.0)

        for level in range(5):
            features.append(bid_px[level] / 100.0 if level < len(bid_px) else 0.0)
            features.append(bid_vol[level] / 100.0 if level < len(bid_vol) else 0.0)
            features.append(ask_px[level] / 100.0 if level < len(ask_px) else 0.0)
            features.append(ask_vol[level] / 100.0 if level < len(ask_vol) else 0.0)

        wap_top = (bid_px[0] * ask_vol[0] + ask_px[0] * bid_vol[0]) / (bid_vol[0] + ask_vol[0] + 1e-8)
        features.append(wap_top / 100.0)

        depth_ratio = bid_vol_total / ask_vol_total
        features.append(np.clip(depth_ratio, 0, 10))

        bid_vwap = np.average(bid_px[:5], weights=bid_vol[:5] + 1e-8) / 100.0
        ask_vwap = np.average(ask_px[:5], weights=ask_vol[:5] + 1e-8) / 100.0
        features.append(bid_vwap)
        features.append(ask_vwap)

        bid_vol_pct = bid_vol[:5] / (bid_vol_total + 1e-8)
        ask_vol_pct = ask_vol[:5] / (ask_vol_total + 1e-8)
        features.extend(bid_vol_pct.tolist())
        features.extend(ask_vol_pct.tolist())

        result = np.array(features, dtype=np.float32)
        assert result.shape[0] == self.N_FEATURES, f"Expected {self.N_FEATURES} features, got {result.shape[0]}"
        return result
