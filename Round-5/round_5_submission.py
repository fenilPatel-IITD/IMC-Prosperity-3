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

np.math = math

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
    @staticmethod
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

class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.signal = Signal.NEUTRAL
        self.curr = 0

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> JSON:
        return self.signal.value

    def load(self, data: JSON) -> None:
        self.signal = Signal(data)


class ImitateStrategy(SignalStrategy):

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        buy_signal_pairs: list[tuple[str, str]],
        sell_signal_pairs: list[tuple[str, str]],
        logic: str = "OR",
        directional: bool = True,
        quantity_mode: str = "default",  # 'default', 'min', 'max', 'const'
        const_quantity: int = 1,
    ) -> None:
        super().__init__(symbol, limit)
        self.buy_signal_pairs = buy_signal_pairs
        self.sell_signal_pairs = sell_signal_pairs
        self.logic = logic.upper()  # Ensure logic is uppercase ('AND' or 'OR')
        self.directional = directional
        self.quantity_mode = quantity_mode
        self.const_quantity = const_quantity
        self.to_buy = limit  # Remaining quantity we can buy
        self.to_sell = limit  # Remaining quantity we can sell
        self.past_trades = deque(maxlen=100)  # Store past trades for directional matching

    def get_high_low(self, state: TradingState, symbol: str) -> tuple[int | None, int | None]:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if len(buy_orders) > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        if len(sell_orders) > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        
        if len(buy_orders) == 0 and len(sell_orders) > 0:
            return (None, popular_sell_price)
        if len(sell_orders) == 0 and len(buy_orders) > 0:
            return (popular_buy_price, None)
        elif len(sell_orders) == len(buy_orders) == 0:
            return None
        return (popular_buy_price, popular_sell_price)

    def match_pair(self, trade: Trade, pair: tuple[str, str]) -> int:

        if isinstance(pair, str):
            if trade.buyer == pair or trade.seller == pair:
                return 1
            return 0
        if pair[1] and pair[0]:
            if trade.buyer == pair[0] and trade.seller == pair[1]:
                return 1
            elif trade.buyer == pair[1] and trade.seller == pair[0]:
                return -1
            return 0
        if not pair[0]:
            if trade.buyer == pair[1]:
                return -1
            elif trade.seller == pair[1]:
                return 1
            return 0
        if not pair[1]:
            if trade.buyer == pair[0]:
                return 1
            elif trade.seller == pair[0]:
                return -1
            return 0
        raise ValueError("Invalid pair")

    def evaluate_signals(self, trade: Trade, signal_pairs: list[tuple[str, str]]) -> bool:

        matches = [self.match_pair(trade, pair) == 1 for pair in signal_pairs]
        if self.logic == "AND":
            return all([any(self.match_pair(t, pair) for t in self.past_trades) for pair in signal_pairs])
        elif self.logic == "OR":
            return any(matches)
        else:
            raise ValueError("Invalid logic. Use 'AND' or 'OR'.")

    def determine_quantity(self, trade: Trade, less_than:int) -> int:

        qty = trade.quantity
        if self.quantity_mode == "default":
            qty = trade.quantity
        elif self.quantity_mode == "min":
            qty = min(trade.quantity, self.const_quantity)
        elif self.quantity_mode == "max":
            qty = max(max(trade.quantity, self.const_quantity), 1) # Ensure at least 1
        elif self.quantity_mode == "const":
            qty = self.const_quantity
        else:
            raise ValueError("Invalid quantity_mode. Use 'default', 'min', 'max', or 'const'.")
        return min(qty, less_than)
    
    def act(self, state: TradingState):
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]

        self.past_trades.extend(trades)

        position = state.position.get(self.symbol, 0)
        best_ask, best_bid = self.get_high_low(state, self.symbol)
        self.to_buy = self.limit - position  # Remaining quantity we can buy
        self.to_sell = self.limit + position  # Remaining quantity we can sell

        if self.curr == 1:
            if state.position.get(self.symbol, 0) != self.limit:
                    self.buy(best_bid, self.to_buy) if best_bid is not None else None
            else:
                self.curr = 0
        elif self.curr == -1:
            if state.position.get(self.symbol, 0) != -self.limit:
                    self.sell(best_ask, self.to_sell) if best_ask is not None else None
            else:
                self.curr = 0

        mid_ = self.get_mid_price(state, self.symbol)
        for t in trades:
            # Check for buy signal pairs
            if self.evaluate_signals(t, self.buy_signal_pairs):
                quantity = self.determine_quantity(t, self.to_buy)
                if quantity > 0 and self.to_buy >= quantity and t.price >= mid_:
                    self.buy(best_bid, quantity) if best_bid is not None else None
                    self.to_buy -= quantity
                    self.curr = 1
                elif quantity > 0 and t.price <= mid_ and self.to_sell >= quantity:
                    self.sell(best_ask, quantity) if best_ask is not None else None
                    self.to_sell -= quantity
                    self.curr = -1
            # Check for sell signal pairs
            elif self.evaluate_signals(t, self.sell_signal_pairs):
                quantity = self.determine_quantity(t, self.to_sell)
                if quantity > 0 and self.to_sell >= quantity and t.price <= mid_:
                    self.sell(best_ask, quantity) if best_ask is not None else None
                    self.to_sell -= quantity
                    self.curr = -1
                elif quantity > 0 and t.price >= mid_ and self.to_buy >= quantity:
                    self.buy(best_bid, quantity) if best_bid is not None else None
                    self.to_buy -= quantity
                    self.curr = 1

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


class HybridResinStrategy:
    def __init__(self):
        self.product = "RAINFOREST_RESIN"
        self.position_limit = 50
        self.soft_position_limit = 0.7  # fraction of limit before soft liquidation
        self.price_alt = 2  # quote distance from fair value
        self.window_size = 10
        self.window = deque()

    def run(self, state) -> List['Order']:
        orders = []
        position = state.position.get(self.product, 0)
        order_depth = state.order_depths.get(self.product, None)
        if not order_depth:
            return orders

        to_buy = self.position_limit - position
        to_sell = self.position_limit + position

        # === Step 1: Look for special opportunities ===

        # Sell if someone is buying at 10008
        if 10008 in order_depth.sell_orders and to_sell > 0:
            vol = -order_depth.sell_orders[10008]
            qty = min(vol, to_sell)
            if qty > 0:
                orders.append(Order(self.product, 10008, -qty))
                to_sell -= qty

        # Buy if someone is selling at 9992
        if 9992 in order_depth.buy_orders and to_buy > 0:
            vol = order_depth.buy_orders[9992]
            qty = min(vol, to_buy)
            if qty > 0:
                orders.append(Order(self.product, 9992, qty))
                to_buy -= qty


        # === Step 2: Passive market making around fair value ===
        fair_value = 10000
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # Update position saturation tracking
        self.window.append(abs(position) == self.position_limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = (
            len(self.window) == self.window_size
            and sum(self.window) >= self.window_size / 2
            and self.window[-1]
        )
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        # Adjust passive quote prices based on inventory pressure
        max_buy_price = fair_value - self.price_alt if position > self.soft_position_limit * self.position_limit else fair_value
        min_sell_price = fair_value + self.price_alt if position < -self.soft_position_limit * self.position_limit else fair_value

        # Place passive buy quotes
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                orders.append(Order(self.product, price, qty))
                to_buy -= qty

        if to_buy > 0 and hard_liquidate:
            orders.append(Order(self.product, fair_value, to_buy // 2))
            to_buy -= to_buy // 2

        if to_buy > 0 and soft_liquidate:
            orders.append(Order(self.product, fair_value - 2, to_buy // 2))
            to_buy -= to_buy // 2

        if to_buy > 0 and buy_orders:
            best_buy_price = max(buy_orders, key=lambda x: x[1])[0]
            price = min(max_buy_price, best_buy_price + 1)
            orders.append(Order(self.product, price, to_buy))

        # Place passive sell quotes
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                orders.append(Order(self.product, price, -qty))
                to_sell -= qty

        if to_sell > 0 and hard_liquidate:
            orders.append(Order(self.product, fair_value, -to_sell // 2))
            to_sell -= to_sell // 2

        if to_sell > 0 and soft_liquidate:
            orders.append(Order(self.product, fair_value + 2, -to_sell // 2))
            to_sell -= to_sell // 2

        if to_sell > 0 and sell_orders:
            best_sell_price = min(sell_orders, key=lambda x: x[1])[0]
            price = max(min_sell_price, best_sell_price - 1)
            orders.append(Order(self.product, price, -to_sell))

        return orders

    def save(self):
        return list(self.window)

    def load(self, data, *args):
        self.window = deque(data)

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int | None:
        order_depth = state.order_depths[self.symbol]
        
        # Top 3 buys and sells, filtering out zero volume
        buys = [(p, v) for p, v in sorted(order_depth.buy_orders.items(), reverse=True)[:3] if v > 0]
        sells = [(p, v) for p, v in sorted(order_depth.sell_orders.items())[:3] if v < 0]

        # Calculate VWAPs only if there are valid volumes
        buy_vwap = sum(p * v for p, v in buys) / sum(v for _, v in buys) if buys else None
        sell_vwap = sum(p * -v for p, v in sells) / sum(-v for _, v in sells) if sells else None

        if buy_vwap is not None and sell_vwap is not None:
            return round((buy_vwap + sell_vwap) / 2)
        return None
 
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

class PicnicBasket1StrategyV2(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.spread = 2.5  # Market making spread
        self.cooldown_ticks = 2
        self.cooldown_long = 1
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
        long_threshold, short_threshold = (-20,40)

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
        best_ask= min(state.order_depths["PICNIC_BASKET1"].sell_orders.keys())
        best_bid= max(state.order_depths["PICNIC_BASKET1"].buy_orders.keys())
        fair_value = (best_ask+best_bid)/2
        if fair_value is None:return
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_price = int(fair_value - self.spread)
        sell_price = int(fair_value + self.spread)

        buy_volume = min(self.limit - position, 20)
        sell_volume = min(self.limit + position, 20)

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
        if(position >= self.limit*0.7):
            return
        to_buy = int((self.limit*0.7 - position))
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        if(position <= -self.limit*0.7):
            return
        to_sell = int((self.limit*0.7 + position))
        if to_sell > 0:
            self.sell(price, to_sell)

class PicnicBasket2StrategyV2(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 4, "JAMS": 2}
        self.window = deque()
        self.window_size = 30 #30
        self.soft_position_limit = 0.6 #0.6

    def act(self, state: TradingState) -> None:
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "PICNIC_BASKET2"]):
            return
        best_ask = min(state.order_depths["PICNIC_BASKET2"].sell_orders.keys())
        best_bid = max(state.order_depths["PICNIC_BASKET2"].buy_orders.keys())
        true_value = (best_ask + best_bid) / 2
        if true_value is None:
            return
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        self.price_alt = 3
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

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
                self.buy(popular_price, min(to_buy, 8))
                to_buy -= min(to_buy, 8)

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
            price = min(max_buy_price, popular_buy_price + 2)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and buy_orders:
            popular_price = max(buy_orders, key=lambda tup: tup[1])[0]
            if popular_price >= min_sell_price:
                self.sell(popular_price, min(to_sell, 8))
                to_sell -= min(to_sell, 8)

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
            price = max(min_sell_price, popular_sell_price - 2)
            self.sell(price, to_sell)

    def save(self):
        return {
            "window": list(self.window),
        }

    def load(self, data, *args) -> None:
        self.args = args
        self.window = deque(data.get("window", []), maxlen=self.window_size)

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

        T = (30000 - state.timestamp / 100) / 10000 / 250
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

class MacronStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, threshold:int = 15, cooldown:int = 15, window_size:int = 50) -> None:
        super().__init__(symbol, limit)
        self.lower_threshold = 43
        self.higher_threshold = 44  # criticalSunlightIndex threshold
        self.fair_a = 8.0998
        self.fair_b = -960.7
        self.previous_sunlightIndex = None
        self.window_size =window_size
        self.window = deque(maxlen = self.window_size)
        self.soft_position_limit = 0.3
        self.price_alt = 2
        self.entered = False
        self.threshold = threshold
        self.cooldown = cooldown
        self.cooldown_timer = 0
    def get_true_value(self, state: TradingState) -> Optional[int]:
        buy_prices = [trade.price for trade in self.caesar_trades if trade.buyer == "Caesar"]
        sell_prices = [trade.price for trade in self.caesar_trades if trade.seller == "Caesar"]

        recent_buys = buy_prices[-1:]
        recent_sells = sell_prices[-1:]

        if not recent_buys or not recent_sells:
            return None

        avg_buy = sum(recent_buys) / len(recent_buys)
        avg_sell = sum(recent_sells) / len(recent_sells)
        return round((avg_buy + avg_sell) / 2)

    def get_fair_value(self, observation) -> float: # type: ignore
        return self.fair_a * observation.sugarPrice + self.fair_b

    def act(self, state: TradingState) -> None:
        # Update Caesar trades

        obs = state.observations.conversionObservations[self.symbol]
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        csi = obs.sunlightIndex

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return

        mid_price = (best_bid + best_ask) / 2
        self.window.append(mid_price)


        if self.previous_sunlightIndex is None:
            self.previous_sunlightIndex = csi
            return

        if csi >= 43 and csi > self.previous_sunlightIndex and self.entered:
            to_buy = self.limit - position
            sell_orders = sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []
            for price, volume in sell_orders:
                if to_buy > 0:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity
            if to_buy == 0:
                self.entered = False

        elif (csi >= self.higher_threshold and csi <= self.previous_sunlightIndex) or (
            csi > self.lower_threshold and csi > self.previous_sunlightIndex
        ):

            if self.cooldown_timer > 0:
                self.cooldown_timer -= 1
                return

            if len(self.window) < self.window.maxlen:
                return

            mean_price = sum(self.window) / len(self.window)
            std = np.std(self.window)
            upper_band = mean_price + self.threshold * std
            lower_band = mean_price - self.threshold * std

            if mid_price > upper_band and position > -self.limit:
                self.sell(mid_price, int((position + self.limit) * 0.8))
                self.cooldown_timer = self.cooldown
            elif mid_price < lower_band and position < self.limit:
                self.buy(mid_price, int((-position + self.limit) * 0.8))
                self.cooldown_timer = self.cooldown

        else:
            self.entered = True
            if self.previous_sunlightIndex > csi:
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders)
                    buy_qty = self.limit - position
                    if buy_qty > 0:
                        self.buy(best_ask, buy_qty)
            elif self.previous_sunlightIndex < csi:
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders)
                    sell_qty = self.limit + position
                    if sell_qty > 0:
                        self.sell(best_bid, sell_qty)

        self.previous_sunlightIndex = csi

    def save(self):
        return {
            "previous_sunlightIndex": self.previous_sunlightIndex,
            "window": list(self.window),
            "entered": self.entered,
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data, *args) -> None:
        self.args = args
        self.previous_sunlightIndex = data.get("previous_sunlightIndex", None)
        self.window = deque(data.get("window", []), maxlen=self.window_size)
        self.entered = data.get("entered", False)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class ZScoreVoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, window: int = 20, max_hist: int = 80,
                 z_entry: float = 1.8, z_exit: float = 0.0) -> None:
        super().__init__(symbol, limit)
        self.window = window
        self.max_hist = max_hist
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.mid_history = deque(maxlen=max_hist)

    def get_swmid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])

        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)

        mid_price = self.get_swmid(order_depth)
        if mid_price is None:
            return

        self.mid_history.append(mid_price)
        if len(self.mid_history) <= self.window:
            return

        mean = np.mean(self.mid_history)
        std = np.std(self.mid_history)
        zscore = (mid_price - mean) / std if std > 0 else (mid_price - mean)

        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        if zscore > self.z_entry and position > -self.limit:
            qty = min(self.limit + position, self.limit)
            self.sell(best_bid, qty)

        elif zscore < -self.z_entry and position < self.limit:
            qty = min(self.limit - position, self.limit)
            self.buy(best_ask, qty)

        elif abs(zscore) <= self.z_exit and position != 0:
            if position > 0:
                self.sell(best_bid, position)
            else:
                self.buy(best_ask, -position)


class VolcanicRockStrategyV2:
    def __init__(self):
        self.product = "VOLCANIC_ROCK"
        self.voucher_1 = "VOLCANIC_ROCK_VOUCHER_9750"
        self.voucher_2 = "VOLCANIC_ROCK_VOUCHER_10250"

        self.history = deque(maxlen=50)
        self.z = 0

        self.strike_prices = {
            self.voucher_1: 9750,
            self.voucher_2: 10250
        }

    def run(self, state: TradingState) -> List[Order]:
        orders = []
        rock_depth = state.order_depths.get(self.product)

        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders)
            rock_ask = min(rock_depth.sell_orders)
            rock_mid = (rock_bid + rock_ask) / 2
            self.history.append(rock_mid)

        if len(self.history) >= 50:
            recent = np.array(self.history)
            mean = np.mean(recent)
            std = np.std(recent)
            self.z = (recent[-1] - mean) / std if std > 0 else 0

            base_threshold = 1.8
            dynamic_threshold = base_threshold + 0.2 * (std / 3)

            position = state.position.get(self.product, 0)
            position_limit = 400

            if self.z < -dynamic_threshold and rock_depth and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders)
                qty = -rock_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    orders.append(Order(self.product, best_ask, buy_qty))

            elif self.z > dynamic_threshold and rock_depth and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders)
                qty = rock_depth.buy_orders[best_bid]
                sell_qty = min(qty, position + position_limit)
                if sell_qty > 0:
                    orders.append(Order(self.product, best_bid, -sell_qty))

        orders += self.hedge_with_vouchers(state)
        return orders

    def hedge_with_vouchers(self, state: TradingState) -> List[Order]:
        orders = []

        position_limit = 200
        rock_price = np.mean(self.history) if self.history else 10200

        pos_rock = state.position.get(self.product, 0)
        pos_1 = state.position.get(self.voucher_1, 0)
        pos_2 = state.position.get(self.voucher_2, 0)

        delta_1 = self.get_delta_estimate(rock_price, self.strike_prices[self.voucher_1])
        delta_2 = self.get_delta_estimate(rock_price, self.strike_prices[self.voucher_2])

        net_delta = (
            pos_rock * 1.0 +
            pos_1 * delta_1 +
            pos_2 * delta_2
        )

        voucher_info = [
            {
                "symbol": self.voucher_1,
                "delta": delta_1,
                "pos": pos_1,
                "depth": state.order_depths.get(self.voucher_1)
            },
            {
                "symbol": self.voucher_2,
                "delta": delta_2,
                "pos": pos_2,
                "depth": state.order_depths.get(self.voucher_2)
            }
        ]

        for v in voucher_info:
            if abs(net_delta) < 0.5 or not v["depth"]:
                continue

            delta = v["delta"]
            pos = v["pos"]
            depth = v["depth"]
            symbol = v["symbol"]

            max_hedge_qty = int(net_delta / delta)
            max_hedge_qty = max(-position_limit - pos, min(position_limit - pos, max_hedge_qty))

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

    def get_delta_estimate(self, underlying_price: float, strike: float, dte: float = 3.5) -> float:
        """Estimate soft delta using a sigmoid-like function."""
        k = 0.05 * dte
        return 1 / (1 + math.exp(-k * (underlying_price - strike)))

    def save(self):
        return {
            "history": list(self.history),
            "z": self.z,
        }

    def load(self, data: dict):
        self.history = deque(data.get("history", []), maxlen=50)
        self.z = data.get("z", 0)


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
            "RAINFOREST_RESIN": HybridResinStrategy(), 
            "KELP": KelpStrategy("KELP", self.limits["KELP"]), 
            "SQUID_INK": SquidInkStrategy("SQUID_INK", self.limits["SQUID_INK"]), 
        
            
            "CROISSANTS": ImitateStrategy(
                symbol="CROISSANTS",
                limit=self.limits["CROISSANTS"],
                buy_signal_pairs=[
                    ("Olivia", "Caesar"),
                ],
                sell_signal_pairs=[
                    ("Caesar", "Olivia"),
                ],
                logic="OR",
                directional=True,
                quantity_mode="max",
                const_quantity=250,
            ),
            
            "PICNIC_BASKET1": PicnicBasket1StrategyV2("PICNIC_BASKET1", self.limits["PICNIC_BASKET1"]), 
            "PICNIC_BASKET2": PicnicBasket2StrategyV2("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"]), 

            "VOLCANIC_ROCK": VolcanicRockStrategyV2(), 

            "VOLCANIC_ROCK_VOUCHER_9500": ZScoreVoucherStrategy("VOLCANIC_ROCK_VOUCHER_9500", self.limits["VOLCANIC_ROCK_VOUCHER_9500"]),
            "VOLCANIC_ROCK_VOUCHER_10000": VolOptionStrategy("VOLCANIC_ROCK_VOUCHER_10000", self.limits["VOLCANIC_ROCK_VOUCHER_10000"]),

            "MAGNIFICENT_MACARONS": MacronStrategy("MAGNIFICENT_MACARONS", self.limits["MAGNIFICENT_MACARONS"], 0.1, 0, 90),   
        }
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
            conversions = 0

            old_data = json.loads(state.traderData) if state.traderData else {}
            new_data = {}
            orders = {
                prod: [] for prod in self.limits.keys()
            }
            new_data = {}

            for symbol, strategy in self.strategies.items():
                if symbol in old_data:
                    strategy.load(old_data[symbol])
                if symbol in state.order_depths:
                    ords = strategy.run(state)
                    for order in ords:
                        orders[order.symbol].append(order)
                new_data[symbol] = strategy.save()
                
            trader_data = json.dumps(new_data, separators=(",", ":"))
            logger.flush(state, orders, conversions, trader_data)
            return orders, conversions, trader_data
