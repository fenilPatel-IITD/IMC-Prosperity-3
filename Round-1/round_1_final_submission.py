import numpy as np
import json
import math
import statistics
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for trade_list in trades.values() for t in trade_list
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {
            p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sunlight, o.humidity]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for order_list in orders.values() for o in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

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

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            self.buy(true_value, to_buy // 2)
            to_buy -= to_buy // 2

        if to_buy > 0 and soft_liquidate:
            self.buy(true_value - 2, to_buy // 2)
            to_buy -= to_buy // 2

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            self.buy(min(max_buy_price, popular_buy_price + 1), to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            self.sell(true_value, to_sell // 2)
            to_sell -= to_sell // 2

        if to_sell > 0 and soft_liquidate:
            self.sell(true_value + 2, to_sell // 2)
            to_sell -= to_sell // 2

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            self.sell(max(min_sell_price, popular_sell_price - 1), to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

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

        return round((buy_vwap + sell_vwap) / 2) if buy_vwap and sell_vwap else 10000

class SquidInkStrategy(Strategy):
    PRODUCT = "SQUID_INK"
    POSITION_LIMIT = 50

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Tunable parameters for mean reversion
        self.MA_WINDOW = 50
        self.ENTRY_THRESHOLD_STD_DEV = 2.0
        self.TRADE_SIZE = 1
        self.TARGET_INVENTORY_LEVEL = 50
        self.NEUTRALIZE_RATE = 1
        self.NEUTRALIZE_WIDTH = 0

        self.price_history = deque(maxlen=self.MA_WINDOW)

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        # Calculate mid price from the best bid and best ask
        best_bid = max(buy_orders.keys()) if buy_orders else None
        best_ask = min(sell_orders.keys()) if sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) // 2
        return 10000  # Default value if no bids/asks available

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)

        # Gather the mid price and update the price history
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            current_mid_price = (best_bid + best_ask) / 2.0
            self.price_history.append(current_mid_price)

        # Compute moving average and standard deviation
        if len(self.price_history) >= self.MA_WINDOW // 2:
            ma = sum(self.price_history) / len(self.price_history)
            variance = sum((p - ma) ** 2 for p in self.price_history) / len(self.price_history)
            std_dev = np.sqrt(variance)

            # Determine upper and lower entry thresholds based on moving average and std dev
            upper_threshold = ma + self.ENTRY_THRESHOLD_STD_DEV * std_dev
            lower_threshold = ma - self.ENTRY_THRESHOLD_STD_DEV * std_dev

            # Set trade size and other parameters
            to_buy = self.limit - current_position
            to_sell = self.limit + current_position

            # Check if price crosses upper or lower threshold
            if current_mid_price > upper_threshold and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                qty_to_sell = min(self.TRADE_SIZE, best_bid_volume)
                if current_position - qty_to_sell >= -self.POSITION_LIMIT:
                    self.sell(best_bid, qty_to_sell)

            elif current_mid_price < lower_threshold and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                qty_to_buy = min(self.TRADE_SIZE, best_ask_volume)
                if current_position + qty_to_buy <= self.POSITION_LIMIT:
                    self.buy(best_ask, qty_to_buy)

            # Neutralizing position if it deviates from the target
            position_after_mr_trades = current_position + to_buy - to_sell
            remaining_buy_capacity = self.POSITION_LIMIT - position_after_mr_trades
            remaining_sell_capacity = self.POSITION_LIMIT + position_after_mr_trades

            if position_after_mr_trades > self.TARGET_INVENTORY_LEVEL and remaining_sell_capacity > 0:
                qty_to_neutralize_max = self.NEUTRALIZE_RATE
                target_neutralize_price = round(ma + self.NEUTRALIZE_WIDTH)
                available_bids = {p: v for p, v in order_depth.buy_orders.items() if p <= target_neutralize_price}
                if available_bids:
                    best_neutralize_bid = max(available_bids.keys())
                    available_volume = available_bids[best_neutralize_bid]
                    sell_amount = min(qty_to_neutralize_max, available_volume, remaining_sell_capacity)
                    if sell_amount > 0:
                        self.sell(best_neutralize_bid, sell_amount)

            elif position_after_mr_trades < -self.TARGET_INVENTORY_LEVEL and remaining_buy_capacity > 0:
                qty_to_neutralize_max = self.NEUTRALIZE_RATE
                target_neutralize_price = round(ma - self.NEUTRALIZE_WIDTH)
                available_asks = {p: abs(v) for p, v in order_depth.sell_orders.items() if p >= target_neutralize_price}
                if available_asks:
                    best_neutralize_ask = min(available_asks.keys())
                    available_volume = available_asks[best_neutralize_ask]
                    buy_amount = min(qty_to_neutralize_max, available_volume, remaining_buy_capacity)
                    if buy_amount > 0:
                        self.buy(best_neutralize_ask, buy_amount)

    def save(self) -> JSON:
        return list(self.price_history)

    def load(self, data: JSON) -> None:
        self.price_history = deque(data, maxlen=self.MA_WINDOW)


class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }
        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
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
                orders[symbol] = strategy.run(state)
            new_data[symbol] = strategy.save()

        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
