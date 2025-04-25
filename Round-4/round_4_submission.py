import numpy as np
import math
import json
from math import log, sqrt, exp, erf
from statistics import *
from datamodel import *
from abc import ABC, abstractmethod
from typing import Deque, Optional, TypeAlias, Any, List
from collections import deque
import statistics
from enum import IntEnum
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Signal(IntEnum):
    NEUTRAL = 0
    SELL = 1
    BUY = 2
    DO_NOTHING = -1
    BUY_SOS = 3
    SELL_SOS = 4

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, *args) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON, *args) -> None:
        self.args = args
        pass

    # @abstractmethod
    def get_mid_price(state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if len(buy_orders) > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        if len(sell_orders) > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        
        if len(buy_orders) == 0 and len(sell_orders) > 0:
            return popular_sell_price
        if len(sell_orders) == 0 and len(buy_orders) > 0:
            return popular_buy_price
        elif len(sell_orders) == len(buy_orders) == 0:
            return None

        return (popular_buy_price + popular_sell_price) / 2

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size:int = 10, soft_position_limit: float = 0.5, price_alt:int = 1) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = window_size
        self.soft_position_limit = soft_position_limit
        self.price_alt = price_alt
        
    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        if true_value is None:
            return

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - self.price_alt if position > self.limit * self.soft_position_limit else true_value
        min_sell_price = true_value + self.price_alt if position < self.limit * -self.soft_position_limit else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON, *args) -> None:
        self.args = args
        self.window = deque(data)

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, *args) -> None:
        super().__init__(symbol, limit)
        self.args = args
        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState, *args) -> None:
        new_signal = self.get_signal(state)
        if new_signal == Signal.DO_NOTHING:
            return
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
            
        buy_price = self.get_buy_price(order_depth)
        sell_price = self.get_sell_price(order_depth)

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                if buy_price is None:
                    return
                self.buy(buy_price, -position)
            elif position > 0:
                if sell_price is None:
                    return
                self.sell(sell_price, position)
        elif self.signal == Signal.SELL:
            if sell_price is None:
                return
            self.sell(sell_price, self.limit + position)
        elif self.signal == Signal.BUY:
            if buy_price is None:
                return
            self.buy(buy_price, self.limit - position)


    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    def save(self) -> JSON:
        return self.signal.value

    def load(self, data: JSON, *args) -> None:
        self.args = args
        self.signal = Signal(data)
        
class MeanReversionStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 10, z_score_threshold: float = 2.0) -> None:
        super().__init__(symbol, limit)
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.prices = deque(maxlen=window_size)

    def get_signal(self, state: TradingState) -> Signal | None:
        order_depth = state.order_depths[self.symbol]
        price = self.get_buy_price(order_depth)

        self.prices.append(price)

        if len(self.prices) < self.window_size:
            return Signal.NEUTRAL

        mean_price = statistics.mean(self.prices)
        std_dev = statistics.stdev(self.prices)

        if std_dev == 0:
            return Signal.NEUTRAL

        z_score = (price - mean_price) / std_dev

        if z_score > self.z_score_threshold:
            return Signal.BUY
        elif z_score < -self.z_score_threshold:
            return Signal.SELL

        return Signal.NEUTRAL
    
    def save(self) -> JSON:
        return {
            "window_size": self.window_size,
            "z_score_threshold": self.z_score_threshold,
            "prices": list(self.prices),}
        
    def load(self, data: JSON, *args) -> None:
        self.args = args
        self.window_size = data["window_size"]
        self.z_score_threshold = data["z_score_threshold"]
        self.prices = deque(data["prices"], maxlen=self.window_size)

class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buys = sorted(order_depth.buy_orders.items(), reverse=True)[:3]
        sells = sorted(order_depth.sell_orders.items())[:3]

        buy_vwap = sum(p * v for p, v in buys) / sum(v for _, v in buys) if buys else 0
        sell_vwap = sum(p * -v for p, v in sells) / sum(-v for _, v in sells) if sells else 0

        return round((buy_vwap + sell_vwap) / 2) if buy_vwap and sell_vwap else None
    
class SquidInkStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size=50):
        """
        :param window_size: Window stores last `window_size` mid prices
        """
        super().__init__(symbol, limit)
        self.window_size = window_size
        self.window = deque(maxlen=window_size)

    def act(self, state: TradingState) -> None:
        self.position = state.position.get(self.symbol, 0)
        base_value = 2000
        
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # Calculate current mid price
        hit_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        hit_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        mid_price = (hit_buy_price + hit_sell_price) / 2
        self.window.append(mid_price)

        # Mean is just simple moving average
        mean = np.mean(self.window)
        
        # Find out how many items we can buy/sell
        pos_window_size = 120
        pos_window_max_var = 200
        pos_window_center = self.limit * (base_value - mean) / pos_window_max_var
        pos_window_bottom = max(-self.limit, pos_window_center - pos_window_size / 2)
        pos_window_top = min(self.limit, pos_window_center + pos_window_size / 2)
        
        to_buy = max(pos_window_top - self.position, 0)
        to_sell = max(-pos_window_bottom + self.position, 0)
        
        inventory_ratio = self.position / self.limit
        if inventory_ratio >= 0:
            sell_limit_factor = max((1 - inventory_ratio) ** 6,0)
            buy_limit_factor = 1 + sell_limit_factor
        else:
            buy_limit_factor = max((1 + inventory_ratio) ** 6,0)
            sell_limit_factor = 1 + buy_limit_factor
        
        buy_buffer = 5
        buy_base_value_diff_factor = 3.75
        buy_weighting = 1 + (buy_base_value_diff_factor * (hit_sell_price / base_value - 1))
        
        # Smaller buy buffer means we buy more!!!
        # buy_weighting < 1 if hit sell price is below base_value (buy more when price below base)
        # buy_limit_factor < 1 if we are negative position (buy more when we are short)
        adj_buy_buffer = buy_buffer * buy_weighting * buy_limit_factor
        best_buy_price = round(mean - adj_buy_buffer)
        
        sell_buffer = 5
        sell_base_value_diff_factor = 3.75
        sell_weighting = 1 - (sell_base_value_diff_factor * (hit_buy_price / base_value - 1))
        
        # Smaller sell buffer means we sell more!!!
        # sell_weighting < 1 if hit sell price is above base_value (sell more when price above base)
        # sell_limit_factor < 1 if we are positive position (sell more when we have shit to sell)
        adj_sell_buffer = sell_buffer * sell_weighting * sell_limit_factor
        best_sell_price = round(mean + adj_sell_buffer)
        
        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(best_buy_price, popular_buy_price + 1)
            self.buy(price, int(to_buy)) 

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(best_sell_price, popular_sell_price - 1)
            self.sell(price, int(to_sell)) 

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data, maxlen=self.window_size)
        
class VolumeWeightedStrategy(KelpStrategy):
    def _init_(self, symbol: str, limit: int) -> None:
        super()._init_(symbol, limit)
        self.recent_trade_window = 30

    def get_true_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]
        
        top_buys = sorted(depth.buy_orders.items(), reverse=True)[:3]
        top_sells = sorted(depth.sell_orders.items())[:3]

        total_buy_vol = sum(v for p, v in top_buys)
        total_sell_vol = sum(v for p, v in top_sells)
        
        weighted_buy = sum(p * v for p, v in top_buys) / total_buy_vol if total_buy_vol > 0 else 0
        weighted_sell = sum(p * v for p, v in top_sells) / total_sell_vol if total_sell_vol > 0 else 0

        fair_value = (weighted_buy + weighted_sell) / 2
        
        return round(fair_value)

class PicnicBasket1Strategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.spread = 60  # Market making spread
        self.cooldown_ticks = 0
        self.cooldown_long = 0
        self.cooldown_short = 0

    def act(self, state: TradingState) -> None:
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return

        # ----------------------
        # Step 1: Arbitrage logic
        # ----------------------
        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        djembe = self.get_mid_price(state, "DJEMBES")
        basket1 = self.get_mid_price(state, "PICNIC_BASKET1")

        diff1 = basket1 - 6 * croissant - 3 * jam - djembe
        long_threshold, short_threshold = {
            "CROISSANTS": (0, 20),
            "JAMS": (-22, 50),
            "DJEMBES": (-10, 35),
            "PICNIC_BASKET1": (-10, 70),
        }[self.symbol]

        if diff1 < long_threshold and self.cooldown_long == 0:
            self.go_long(state)
            self.cooldown_long = self.cooldown_ticks
        elif diff1 > short_threshold and self.cooldown_short == 0:
            self.go_short(state)
            self.cooldown_short = self.cooldown_ticks

        self.cooldown_long = max(0, self.cooldown_long - 1)
        self.cooldown_short = max(0, self.cooldown_short - 1)

        # ----------------------
        # Step 2: Market Making
        # ----------------------
        fair_value = 6 * croissant + 3 * jam + djembe
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_price = int(fair_value - self.spread + 10)
        sell_price = int(fair_value + self.spread + 10)

        buy_volume = min(self.limit - position, 2)
        sell_volume = min(self.limit + position, 2)

        if buy_volume > 0 and self.cooldown_long == 0:
            self.buy(buy_price, buy_volume)

        if sell_volume > 0 and self.cooldown_short == 0:
            self.sell(sell_price, sell_volume)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        if(position >= self.limit*0.8):
            return
        to_buy = int((self.limit*0.7 - position))
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        if(position <= -self.limit*0.8):
            return
        to_sell = int((self.limit*0.7 + position))
        if to_sell > 0:
            self.sell(price, to_sell)

class PicnicBasket2Strategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 4, "JAMS": 2}
        self.spread = 30  # Market making spread
        self.cooldown_ticks = 0
        self.cooldown_long = 0
        self.cooldown_short = 0

    def act(self, state: TradingState) -> None:
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "PICNIC_BASKET2"]):
            return

        # ----------------------
        # Step 1: Arbitrage logic
        # ----------------------
        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        basket2 = self.get_mid_price(state, "PICNIC_BASKET2")

        diff2 = basket2 - 4 * croissant - 2 * jam
        long_threshold, short_threshold = (-47, 40)

        if diff2 < long_threshold and self.cooldown_long == 0:
            self.go_long(state)
            self.cooldown_long = self.cooldown_ticks
        elif diff2 > short_threshold and self.cooldown_short == 0:
            self.go_short(state)
            self.cooldown_short = self.cooldown_ticks

        self.cooldown_long = max(0, self.cooldown_long - 1)
        self.cooldown_short = max(0, self.cooldown_short - 1)

        # ----------------------
        # Step 2: Market Making
        # ----------------------
        fair_value = 4 * croissant + 2 * jam
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_price = int(fair_value - self.spread + 10)
        sell_price = int(fair_value + self.spread + 10)

        buy_volume = min(self.limit - position, 2)
        sell_volume = min(self.limit + position, 2)

        if buy_volume > 0 and self.cooldown_long == 0:
            self.buy(buy_price, buy_volume)

        if sell_volume > 0 and self.cooldown_short == 0:
            self.sell(sell_price, sell_volume)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        if position >= self.limit * 0.7:
            return
        to_buy = int((self.limit * 0.7 - position))
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        if position <= -self.limit * 0.7:
            return
        to_sell = int((self.limit * 0.7 + position))
        if to_sell > 0:
            self.sell(price, to_sell)

class Basket1ProductsStrategy(Strategy):
    def act(self, state: TradingState) -> None:
        if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return

        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        djembe = self.get_mid_price(state, "DJEMBES")
        basket1 = self.get_mid_price(state, "PICNIC_BASKET1")

        diff1 = basket1 - 6 * croissant - 3 * jam - djembe

        long_threshold, short_threshold = {
            "CROISSANTS": (-10, 70),
            "JAMS": (-22, 50),
            "DJEMBES": (-10, 35),
        }[self.symbol]

        if diff1 < long_threshold:
            self.go_long(state)
        elif diff1 > short_threshold:
            self.go_short(state)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)

class Basket2ProductsStrategy(Strategy):
    def act(self, state: TradingState) -> None:
        if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return

        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        basket1 = self.get_mid_price(state, "PICNIC_BASKET2")

        diff1 = basket1 - 4 * croissant - 2 * jam 

        long_threshold, short_threshold = {
            "CROISSANTS": (-40, 35),
            "DJEMBES": (-30, 40),
        }[self.symbol]

        if diff1 < long_threshold:
            self.go_short(state)
        elif diff1 > short_threshold:
            self.go_long(state)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)

# RSI Strategy using mean reversion
class RSIStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 14, overbought: float = 70, oversold: float = 30) -> None:
        super().__init__(symbol, limit)
        self.window_size = window_size
        self.overbought = overbought
        self.oversold = oversold
        self.gains = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)

    def get_signal(self, state: TradingState) -> Signal | None:
        order_depth = state.order_depths[self.symbol]
        price = self.get_buy_price(order_depth)

        if len(self.gains) + len(self.losses) > 0:
            prev_price = self.gains[-1] + self.losses[-1] if self.gains and self.losses else price
            change = price - prev_price
            self.gains.append(max(change, 0))
            self.losses.append(abs(min(change, 0)))

        if len(self.gains) < self.window_size or len(self.losses) < self.window_size:
            return Signal.NEUTRAL

        avg_gain = sum(self.gains) / self.window_size
        avg_loss = sum(self.losses) / self.window_size

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # print(f"RSI: {rsi}, Gains: {list(self.gains)}, Losses: {list(self.losses)}", file="idk.txt")

        if rsi > self.overbought:
            return Signal.SELL
        elif rsi < self.oversold:
            return Signal.BUY

        return Signal.NEUTRAL

    def save(self) -> JSON:
        return {
            "window_size": self.window_size,
            "overbought": self.overbought,
            "oversold": self.oversold,
            "gains": list(self.gains),
            "losses": list(self.losses),
        }

    def load(self, data: JSON, *args) -> None:
        self.args = args
        self.window_size = data["window_size"]
        self.overbought = data["overbought"]
        self.oversold = data["oversold"]
        self.gains = deque(data["gains"], maxlen=self.window_size)
        self.losses = deque(data["losses"], maxlen=self.window_size)

class TrendFollowingStrategy(Strategy):
    """
    You can pass a tuple in thresholds 0 -> upper band, 1 -> lower band
    """
    def __init__(self, symbol: Symbol, limit: int, window: int = 50, threshold: float = 10.0,
                 bias: int = 0, cooldown: int = 10) -> None:
        super().__init__(symbol, limit)
        self.window = window
        self.threshold = threshold
        self.bias = bias
        self.cooldown = cooldown

        self.prices: Deque[float] = deque(maxlen=window)
        self.cooldown_timer = 0

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return
        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        if len(self.prices) < self.window:
            return  # Not enough data

        mean_price = sum(self.prices) / self.window

        # Define bands
        if isinstance(self.threshold, tuple):
            upper_band = mean_price + self.threshold[0]
            lower_band = mean_price - self.threshold[1]
        else:
            upper_band = mean_price + self.threshold
            lower_band = mean_price - self.threshold

        # Mean reversion logic
        if mid_price < lower_band and position < self.limit:
            self.buy(mid_price, self.limit - position)
            self.cooldown_timer = self.cooldown

        elif mid_price > upper_band and position > -self.limit:
            self.sell(mid_price, position + self.limit)
            self.cooldown_timer = self.cooldown

        # Optional bias-based trading
        elif self.bias > 0:
            self.buy(mid_price, min(self.bias, self.limit - position))
        elif self.bias < 0:
            self.sell(mid_price, min(abs(self.bias), position + self.limit))

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class PFTrendHybridStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, box_size: float = 10.0, reversal: int = 3,
                 window: int = 50, cooldown: int = 10):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.cooldown = cooldown

        self.prices: Deque[float] = deque(maxlen=self.window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                # Reversal
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return  # Not enough data yet

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        trend = self.detect_trend(mid_price)

        if trend == 'up' and position < self.limit:
            self.buy(mid_price, self.limit - position)
            self.cooldown_timer = self.cooldown
        elif trend == 'down' and position > -self.limit:
            self.sell(mid_price, position + self.limit)
            self.cooldown_timer = self.cooldown

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class PFTrendHybridStrategyV2(Strategy):
    """
    Hybrid of Point & Figure trend detection and configurable trend-following strategy.
    All parameters from TrendFollowingStrategy are supported.
    """
    def __init__(self, symbol: Symbol, limit: int, box_size: float = 10.0, reversal: int = 3,
                 window: int = 50, threshold: float | tuple = 10.0, bias: int = 0, cooldown: int = 10):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.threshold = threshold
        self.bias = bias
        self.cooldown = cooldown

        self.prices: Deque[float] = deque(maxlen=window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                # Reversal
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return  # Not enough data yet

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        mean_price = sum(self.prices) / self.window
        if isinstance(self.threshold, tuple):
            upper_band = mean_price + self.threshold[0]
            lower_band = mean_price - self.threshold[1]
        else:
            upper_band = mean_price + self.threshold
            lower_band = mean_price - self.threshold

        trend = self.detect_trend(mid_price)

        if trend == 'up':
            if mid_price < upper_band and position < self.limit:
                self.buy(mid_price, self.limit - position)
                self.cooldown_timer = self.cooldown
        elif trend == 'down':
            if mid_price > lower_band and position > -self.limit:
                self.sell(mid_price, position + self.limit)
                self.cooldown_timer = self.cooldown
        else:
            # Optional fallback: bias-based trading
            if self.bias > 0:
                self.buy(mid_price, min(self.bias, self.limit - position))
            elif self.bias < 0:
                self.sell(mid_price, min(abs(self.bias), position + self.limit))

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class PFTrendHybridStrategyV3(Strategy):
    """
    Advanced hybrid strategy with:
    - Trend following during breakouts
    - Mean reversion in choppy zones
    - Market making during neutral/low-volatility periods
    - Bias trading in neutral zones with cooldown
    """
    def __init__(
        self, symbol: Symbol, limit: int,
        box_size: float = 10.0, reversal: int = 3, window: int = 50,
        trend_threshold: float = 10.0, reversion_threshold: float = 2.0,
        vol_window: int = 20, vol_threshold: float = 5.0,
        mm_spread: float = 1.0, mm_size: int = 1,
        cooldown: int = 10, bias_cooldown: int = 2,
        bias: int = 0
    ):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.trend_threshold = trend_threshold
        self.reversion_threshold = reversion_threshold
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.mm_spread = mm_spread
        self.mm_size = mm_size
        self.cooldown = cooldown
        self.bias_cooldown = bias_cooldown
        self.bias = bias

        self.prices: Deque[float] = deque(maxlen=window)
        self.returns: Deque[float] = deque(maxlen=vol_window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0
        self.bias_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def compute_volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        return np.std(self.returns)

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) >= 2:
            self.returns.append(self.prices[-1] - self.prices[-2])

        if len(self.prices) < self.window:
            return

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        mean_price = sum(self.prices) / self.window
        z_score = (mid_price - mean_price) / (np.std(self.prices) + 1e-6)
        vol = self.compute_volatility()
        trend = self.detect_trend(mid_price)

        # --- STRATEGY SELECTION LOGIC ---
        if vol > self.vol_threshold:
            self.market_make(mid_price, position)
            return

        if abs(z_score) > self.reversion_threshold:
            # Mean reversion mode
            if z_score > 0 and position > -self.limit:
                self.sell(mid_price, min(self.limit + position, self.limit))
                self.cooldown_timer = self.cooldown
            elif z_score < 0 and position < self.limit:
                self.buy(mid_price, min(self.limit - position, self.limit))
                self.cooldown_timer = self.cooldown
            return

        if trend == 'up' and position < self.limit:
            self.buy(mid_price, self.limit - position)
            self.cooldown_timer = self.cooldown
            self.bias_timer = 0
        elif trend == 'down' and position > -self.limit:
            self.sell(mid_price, position + self.limit)
            self.cooldown_timer = self.cooldown
            self.bias_timer = 0
        else:
            # Neutral zone: bias trading + market making
            if self.bias_timer == 0:
                if self.bias > 0 and position < self.limit:
                    self.buy(mid_price, min(self.bias, self.limit - position))
                elif self.bias < 0 and position > -self.limit:
                    self.sell(mid_price, min(abs(self.bias), position + self.limit))
                self.bias_timer = self.bias_cooldown
            else:
                self.bias_timer -= 1

            self.market_make(mid_price, position)

    def market_make(self, mid_price: float, position: int):
        bid = mid_price - self.mm_spread / 2
        ask = mid_price + self.mm_spread / 2
        if position + self.mm_size <= self.limit:
            self.buy(bid, self.mm_size)
        if position - self.mm_size >= -self.limit:
            self.sell(ask, self.mm_size)

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "returns": list(self.returns),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer,
            "bias_timer": self.bias_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.returns = deque(data.get("returns", []), maxlen=self.vol_window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)
        self.bias_timer = data.get("bias_timer", 0)

class SpecialStrategy(Strategy):
    ...

class Special(SpecialStrategy):
    def buy(self, prod, price, qty):
        l = self.orders.get(prod, [])
        l.append(Order(prod, price, qty))
        self.orders[prod] = l
    def sell(self, prod, price, qty):
        l = self.orders.get(prod, [])
        l.append(Order(prod, price, -qty))
        self.orders[prod] = l
    def __init__(self, symbol: Symbol, limit: int, window: int = 50, threshold: float = 10.0) -> None:
        super().__init__(symbol, limit)
        self.window = window
        self.threshold = threshold
        self.vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            # "VOLCANIC_ROCK_VOUCHER_10000",
            # "VOLCANIC_ROCK_VOUCHER_10250",
            # "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        self.prices: Deque[float] = deque(maxlen=window)

    def act(self, state: TradingState) -> None:
        self.orders={}
        mid_price1 = Strategy.get_mid_price(state, self.symbol)
        if mid_price1 is None:
            return
        self.prices.append(mid_price1)

        if len(self.prices) < self.window:
            return

        mean_price = sum(self.prices) / self.window

        if isinstance(self.threshold, tuple):
            upper_band = mean_price + self.threshold[0]
            lower_band = mean_price - self.threshold[1]
        else:
            upper_band = mean_price + self.threshold
            lower_band = mean_price - self.threshold

        if mid_price1 < lower_band:
            for voucher in self.vouchers:
                mp = Strategy.get_mid_price(state, voucher)
                if mp is None:
                    continue
                position = state.position.get(voucher, 0)
                volume = self.limit - position
                if volume > 0:
                    self.buy(voucher, round(mp), volume)

        elif mid_price1 > upper_band:
            for voucher in self.vouchers:
                mp = Strategy.get_mid_price(state, voucher)
                if mp is None:
                    continue
                position = state.position.get(voucher, 0)
                volume = position + self.limit
                if volume > 0:
                    self.sell(voucher, round(mp), volume)

    def save(self) -> dict:
        return {"prices": list(self.prices)}

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        
np.math = math

class VolOptionStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.implied_vol_history: Dict[str, List[float]] = {
            "VOLCANIC_ROCK_VOUCHER_10000": [],
        }
        self.pos_limit = {
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
        }
        self.ma = {
            "VOLCANIC_ROCK_VOUCHER_10000": 30,
        }

    def act(self, state: TradingState) -> None:
        self.orders.extend(self.fair_value_trading_orders(state))

    def fair_value_trading_orders(self, state) -> List[Order]:
        orders = []
        if "VOLCANIC_ROCK" in state.order_depths:
            current_stock_price = self.get_mid_price(state, "VOLCANIC_ROCK")
        else:
            return orders

        T = (40000 - state.timestamp / 100) / 10000 / 250
        r = 1e-6 # discuss

        for K in [10000]:
            option = f"VOLCANIC_ROCK_VOUCHER_{K}"
            if option not in state.order_depths:
                continue

            current_option_price = self.get_mid_price(state, option)
            if current_option_price is None or current_stock_price is None:
                continue

            implied_vol = self.calculate_implied_volatility(
                current_stock_price, current_option_price, K, T, r
            )
            if not np.isnan(implied_vol):
                self.implied_vol_history[option].append(implied_vol)

            if len(self.implied_vol_history[option]) >= self.ma[option]:
                predicted_vol = self.predict_next_value(
                    self.implied_vol_history[option][-self.ma[option]:]
                )
                fair_value = self.call_option_price(current_stock_price, K, predicted_vol, T, r)

                if fair_value is not None:
                    sorted_asks = sorted(state.order_depths[option].sell_orders.items())
                    sorted_bids = sorted(state.order_depths[option].buy_orders.items(), reverse=True)
                    position = state.position.get(option, 0)

                    taken_asks = 0
                    for ask_price, ask_quantity in sorted_asks:
                        if ask_price < fair_value:
                            qty = min(-ask_quantity, self.pos_limit[option] - position - taken_asks)
                            if qty > 0:
                                orders.append(Order(option, ask_price, qty))
                                taken_asks += qty

                    taken_bids = 0
                    for bid_price, bid_quantity in sorted_bids:
                        if bid_price > fair_value:
                            qty = min(bid_quantity, self.pos_limit[option] + position - taken_bids)
                            if qty > 0:
                                orders.append(Order(option, bid_price, -qty))
                                taken_bids += qty

            if len(self.implied_vol_history[option]) > 300:
                self.implied_vol_history[option].pop(0)

        return orders

    def calculate_implied_volatility(self, S, option_price, K, T, r, max_iterations=300, tolerance=1e-5):
        intrinsic_value = max(0, S - K * np.exp(-r * T))
        if option_price < intrinsic_value or option_price >= S:
            return np.nan

        low_vol, high_vol = 0.001, 0.3
        volatility = 0.1 if option_price / S > 0.1 else 0.05

        for _ in range(max_iterations):
            estimated_price = self.call_option_price(S, K, volatility, T, r)
            diff = estimated_price - option_price

            if abs(diff) < tolerance:
                return volatility

            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility

            volatility = (low_vol + high_vol) / 2.0

            if high_vol - low_vol < tolerance:
                return volatility

        return np.nan

    def call_option_price(self, S, K, sigma, T=1, r=0):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.norm(d1) - K * np.exp(-r * T) * self.norm(d2)

    def norm(self, x):
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

    def predict_next_value(self, data):
        data = np.array(data)
        mask = ~np.isnan(data)
        if not np.any(mask):
            return np.nan

        valid_data = data[mask]
        valid_indices = np.arange(len(data))[mask]
        if len(valid_data) < 2:
            return valid_data[0] if len(valid_data) == 1 else np.nan

        x = valid_indices
        y = valid_data
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return intercept + slope * len(data)

    def save(self) -> dict:
        return {
            "implied_vol_history": {
                k: v[-300:] for k, v in self.implied_vol_history.items()
            }
        }

    def load(self, data: dict, *args) -> None:
        if "implied_vol_history" in data:
            for k, v in data["implied_vol_history"].items():
                self.implied_vol_history[k] = v

    @staticmethod
    def get_mid_price(state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if len(buy_orders) > 0:
            best_bid = buy_orders[0][0]
        else:
            best_bid = None
        if len(sell_orders) > 0:
            best_ask = sell_orders[0][0]
        else:
            best_ask = None

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask
        return None

# Assuming Order and TradingState are defined elsewhere in your codebase
class VolcanicRockMeanReversionStrategy(Strategy):
    def __init__(self, symbol: str = "VOLCANIC_ROCK", position_limit: int = 400, z_threshold: float = 2.2):
        self.symbol = symbol
        self.position_limit = position_limit
        self.z_threshold = z_threshold
        self.history = deque(maxlen=50)
        self.z_score = None


    def update_history(self, state) -> None:
        """
        Update internal price history based on current order book.
        """
        order_depth = state.order_depths.get(self.symbol)
        if order_depth and order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
            self.history.append(mid_price)

            if len(self.history) == self.history.maxlen:
                mean = np.mean(self.history)
                std = np.std(self.history)
                self.z_score = (self.history[-1] - mean) / std if std > 0 else 0

    def act(self, state) -> List:
        """
        Return trading decisions for VOLCANIC_ROCK based on Z-score.
        """
        self.update_history(state)
        orders = []

        if len(self.history) < self.history.maxlen or self.z_score is None:
            return orders  # Not enough data yet

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return orders

        if self.z_score < -self.z_threshold and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            qty = -order_depth.sell_orders[best_ask]
            buy_qty = min(qty, self.position_limit - position)
            if buy_qty > 0:
                orders.append(Order(self.symbol, best_ask, buy_qty))

        elif self.z_score > self.z_threshold and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            qty = order_depth.buy_orders[best_bid]
            sell_qty = min(qty, position + self.position_limit)
            if sell_qty > 0:
                orders.append(Order(self.symbol, best_bid, -sell_qty))
        
        self.orders = orders

        return orders


class VolcanicRockStrategy:  #best till now
    def __init__(self):
        self.product = "VOLCANIC_ROCK"
        self.voucher_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
        self.voucher_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
        self.history = deque([
            10217.5, 10216.5, 10217.0, 10216.0, 10215.5,
            10216.0, 10218.5, 10219.0, 10218.0, 10217.0,
            10216.5, 10217.5, 10218.0, 10217.0, 10216.5,
            10217.0, 10216.0, 10215.0, 10215.5, 10216.5,
            10217.5, 10218.5, 10218.0, 10217.0, 10216.5,
            10216.0, 10215.5, 10216.0, 10217.0, 10218.0,
            10218.5, 10219.5, 10219.0, 10218.0, 10217.5,
            10217.0, 10216.5, 10216.0, 10216.5, 10217.5,
            10218.0, 10218.5, 10218.0, 10217.5, 10217.0,
            10216.0, 10215.5, 10215.0, 10214.5, 10215.5
        ], maxlen=50)
        self.z = 0

    def run(self, state: TradingState) -> List[Order]:
        orders = []
        rock_depth = state.order_depths.get(self.product)

        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders)
            rock_ask = min(rock_depth.sell_orders)
            rock_mid = (rock_bid + rock_ask) / 2
            self.history.append(rock_mid)

        if len(self.history) >= 50:
            recent = np.array(self.history)[-50:]
            mean = np.mean(recent)
            std = np.std(recent)
            self.z = (recent[-1] - mean) / std if std > 0 else 0

            threshold = 2.1
            position = state.position.get(self.product, 0)
            position_limit = 400

            # Buy signal
            if self.z < -threshold and rock_depth and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders)
                qty = -rock_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    orders.append(Order(self.product, best_ask, buy_qty))

            # Sell signal
            elif self.z > threshold and rock_depth and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders)
                qty = rock_depth.buy_orders[best_bid]
                sell_qty = min(qty, position + position_limit)
                if sell_qty > 0:
                    orders.append(Order(self.product, best_bid, -sell_qty))

        # --- Delta Hedge ---
        hedge_orders = self.hedge_with_vouchers(state)
        orders += hedge_orders

        return orders

    def hedge_with_vouchers(self, state: TradingState) -> List[Order]:
        orders = []

        # Constants
        delta_rock = 1
        delta_10250 = 0.2
        delta_10500 = 0.1   
        limit = 200

        # Positions
        pos_rock = state.position.get(self.product, 0)
        pos_10250 = state.position.get(self.voucher_10250, 0)
        pos_10500 = state.position.get(self.voucher_10500, 0)

        # Portfolio delta
        net_delta = (
            pos_rock * delta_rock +
            pos_10250 * delta_10250 +
            pos_10500 * delta_10500
        )

        # Want to reduce delta to ~0
        hedge_depths = [
            (self.voucher_10250, delta_10250, state.order_depths.get(self.voucher_10250), pos_10250),
            (self.voucher_10500, delta_10500, state.order_depths.get(self.voucher_10500), pos_10500)
        ]

        for symbol, delta, depth, pos in hedge_depths:
            if abs(net_delta) < 0.5 or not depth:
                continue

            max_hedge_qty = int(net_delta / delta)

            # Clamp to position limits
            max_hedge_qty = max(-limit - pos, min(limit - pos, max_hedge_qty))

            if max_hedge_qty > 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders)
                ask_qty = -depth.sell_orders[best_ask]
                qty = min(max_hedge_qty, ask_qty)
                if qty > 0:
                    orders.append(Order(symbol, best_ask, qty))
                    net_delta -= qty * delta

            elif max_hedge_qty < 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders)
                bid_qty = depth.buy_orders[best_bid]
                qty = min(-max_hedge_qty, bid_qty)
                if qty > 0:
                    orders.append(Order(symbol, best_bid, -qty))
                    net_delta += qty * delta

        return orders
    
    def save(self):
        pass
    
    def load(self, data: dict):
        pass


class MacronStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.lower_threshold = 43
        self.higher_threshold = 44  # criticalSunlightIndex threshold
        self.fair_a = 3.4464
        self.fair_b = -51.3392
        self.previous_sunlightIndex = None
        self.window = deque()
        self.window_size = 10
        self.soft_position_limit = 0.3
        self.price_alt = 2
        self.entered=False

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
    
        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        # Dynamic volume filter based on order book averages
        avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
        avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
        vol = math.floor(min(avg_sell_volume, avg_buy_volume))

        if (
            len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol]) == 0
            or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol]) == 0
        ):
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return round(mid_price)
        else:
            best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol])
            best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol])
            mid_price = (best_ask + best_bid) / 2
            return round(mid_price)

    def get_fair_value(self, observation: ConversionObservation) -> float:
        return self.fair_a * observation.sugarPrice + self.fair_b

    def act(self, state: TradingState) -> None:
        obs = state.observations.conversionObservations[self.symbol]
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        csi = obs.sunlightIndex
        true_value = self.get_fair_value(obs)
        # true_value=self.get_true_value(state)
        self.price_alt = max(1, int(0.05 * true_value))

        # First time — no previous value to compare
        if self.previous_sunlightIndex is None:
            self.previous_sunlightIndex = csi
            return
        
        if csi>=43 and csi>self.previous_sunlightIndex and self.entered:
            logger.print(f"Entered: {self.entered}")
            to_buy = self.limit - position
            sell_orders = sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []
            for price, volume in sell_orders:
                if to_buy > 0:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity
            if to_buy==0:self.entered=False
        
        elif (csi >= self.higher_threshold and csi<=self.previous_sunlightIndex) or (csi > self.lower_threshold and csi > self.previous_sunlightIndex):
            # Above threshold: regression-based trading
            self.window.append(abs(position) == self.limit)
            if len(self.window) > self.window_size:
                self.window.popleft()

            soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
            hard_liquidate = len(self.window) == self.window_size and all(self.window)

            max_buy_price = true_value - self.price_alt if position > self.limit * self.soft_position_limit else true_value
            min_sell_price = true_value + self.price_alt if position < self.limit * -self.soft_position_limit else true_value

            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True) if order_depth.buy_orders else []
            sell_orders=sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []

            to_buy=self.limit-position
            to_sell=self.limit+position

            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity

            if to_buy > 0 and sell_orders:
                popular_price = max(sell_orders, key=lambda tup: tup[1])[0]
                if popular_price <= max_buy_price:
                    self.buy(popular_price, min(to_buy, 4))
                    to_buy -= min(to_buy, 4)

            if to_buy > 0 and hard_liquidate:
                quantity = to_buy // 2
                self.buy(true_value, quantity)
                to_buy -= quantity

            if to_buy > 0 and soft_liquidate:
                quantity = to_buy // 2
                self.buy(true_value - 2, quantity)
                to_buy -= quantity

            if to_buy > 0 and buy_orders:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                self.buy(price, to_buy)

            # if to_buy > 0:
            #     self.buy(true_value + 1, to_buy)

            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    quantity = min(to_sell, volume)
                    self.sell(price, quantity)
                    to_sell -= quantity

            if to_sell > 0 and buy_orders:
                popular_price = max(buy_orders, key=lambda tup: tup[1])[0]
                if popular_price >= min_sell_price:
                    self.sell(popular_price, min(to_sell, 4))
                    to_sell -= min(to_sell, 4)

            if to_sell > 0 and hard_liquidate:
                quantity = to_sell // 2
                self.sell(true_value, quantity)
                to_sell -= quantity

            if to_sell > 0 and soft_liquidate:
                quantity = to_sell // 2
                self.sell(true_value + 2, quantity)
                to_sell -= quantity

            if to_sell > 0 and sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(price, to_sell)

            # if to_sell > 0:
            #     self.sell(true_value - 1, to_sell)
        else:
            self.entered=True
            # Below threshold: track sunlight trend
            if self.previous_sunlightIndex > csi:
                # Sunlight dropping → BUY aggressively
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders)
                    buy_qty = self.limit - position
                    if buy_qty > 0:
                        self.buy(best_ask, buy_qty)
            elif self.previous_sunlightIndex < csi:
                # Sunlight rising → SELL if price high
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders)
                    sell_qty = self.limit + position
                    if sell_qty > 0:
                        self.sell(best_bid, sell_qty)

        # Update history
        self.previous_sunlightIndex = csi

    def save(self):
        return {
            "previous_sunlightIndex": self.previous_sunlightIndex,
            "window": list(self.window),
            "entered":self.entered
        }

    def load(self, data, *args) -> None:
        self.args = args
        self.previous_sunlightIndex = data.get("previous_sunlightIndex", None)
        self.window = deque(data.get("window", []), maxlen=self.window_size)
        self.entered=data.get("entered",False)


class Trader:
    def __init__(self) -> None:
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,   
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }
        

        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", self.limits["RAINFOREST_RESIN"], 10, 0.75, 1), #final
            "KELP": KelpStrategy("KELP", self.limits["KELP"]), #final
            "SQUID_INK": SquidInkStrategy("SQUID_INK", self.limits["SQUID_INK"]), #final
        
            # "CROISSANTS": TrendFollowingStrategy("CROISSANTS", self.limits["CROISSANTS"], 55, (7,3), -2, 0), #best
            # "JAMS": Basket1ProductsStrategy("JAMS", self.limits["JAMS"]),  #best
            # "DJEMBES": TrendFollowingStrategy("DJEMBES", self.limits["DJEMBES"], 100, 12.5, 0, 25), #best
            
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", self.limits["PICNIC_BASKET1"]), #best
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"]), #best

            "VOLCANIC_ROCK_VOUCHER_9500": Special("VOLCANIC_ROCK_VOUCHER_9500", self.limits["VOLCANIC_ROCK_VOUCHER_9500"], 80, (1,6)), #best
            "VOLCANIC_ROCK_VOUCHER_10000": VolOptionStrategy("VOLCANIC_ROCK_VOUCHER_10000", self.limits["VOLCANIC_ROCK_VOUCHER_10000"]), #best
            "VOLCANIC_ROCK": VolcanicRockStrategy(), #best
            
            "MAGNIFICENT_MACARONS": MacronStrategy("MAGNIFICENT_MACARONS", self.limits["MAGNIFICENT_MACARONS"]),

            # "VOLCANIC_ROCK": VolcanicRockMeanReversionStrategy("VOLCANIC_ROCK", self.limits["VOLCANIC_ROCK"]), #king james original


            # "VOLCANIC_ROCK": VolcanicRockStrategy("VOLCANIC_ROCK", self.limits["VOLCANIC_ROCK"], 2, 1, 1), #best
            # "VOLCANIC_ROCK": PFTrendHybridStrategyV2("VOLCANIC_ROCK", self.limits["VOLCANIC_ROCK"], 3, 2, 90, (2, 4), -2, 0), #best
            # "VOLCANIC_ROCK_VOUCHER_9500": TrendFollowingStrategy("VOLCANIC_ROCK_VOUCHER_9500", self.limits["VOLCANIC_ROCK_VOUCHER_9500"], 80, (1, 6), 0, 0),
            # "VOLCANIC_ROCK_VOUCHER_9750": TrendFollowingStrategy("VOLCANIC_ROCK_VOUCHER_9750", self.limits["VOLCANIC_ROCK_VOUCHER_9750"], 90, (4, 7), 0, 2),
            # "VOLCANIC_ROCK_VOUCHER_10000": TrendFollowingStrategy("VOLCANIC_ROCK_VOUCHER_10000", self.limits["VOLCANIC_ROCK_VOUCHER_10000"], 80, (1, 6), -1, 0),
            # "VOLCANIC_ROCK_VOUCHER_10250": PFTrendHybridStrategyV3("VOLCANIC_ROCK_VOUCHER_10250", self.limits["VOLCANIC_ROCK_VOUCHER_10250"], 8, 2, 90, 6, 1, 100, 6, 4, 2, 3, 0, -1),
            # "VOLCANIC_ROCK_VOUCHER_10500": TrendFollowingStrategy("VOLCANIC_ROCK_VOUCHER_10500", self.limits["VOLCANIC_ROCK_VOUCHER_10500"], 700, 0.5, 0, 0),
        }
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0

        old_data = json.loads(state.traderData) if state.traderData else {}
        new_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_data:
                strategy.load(old_data[symbol])
            if symbol in state.order_depths:
                if isinstance(strategy, SpecialStrategy):
                    ords = strategy.run(state)
                    for k, v in ords.items():
                        if k not in orders:
                            orders[k] = []
                        orders[k].extend(v)
                else:
                    orders[symbol] = strategy.run(state)
            new_data[symbol] = strategy.save()

        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data