import numpy as np
import json
import math
import statistics
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import jsonpickle

# JSON= dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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

    def save(self):
        return None

    def load(self, data) -> None:
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

    def save(self):
        return list(self.window)

    def load(self, data) -> None:
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
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.LIMIT = 50
        self.window_size = 10
        self.ink_history = deque(maxlen=self.window_size)
        self.last_price = None

    def get_fair_value(self, state: TradingState, traderObject: dict) -> float:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= 15
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("SQUID_INK_last_price") is None else traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price") is not None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                if len(self.ink_history) >= self.window_size:
                    returns = np.diff(list(self.ink_history)) / list(self.ink_history)[:-1]
                    volatility = np.std(returns)
                    dynamic_beta = -0.15 * (1 + volatility * 100)
                    pred_returns = last_returns * dynamic_beta
                else:
                    pred_returns = last_returns * -0.15
                fair = mmmid_price + mmmid_price * pred_returns
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return 10000

    def act(self, state: TradingState) -> None:
        import jsonpickle
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            self.ink_history.append((best_bid + best_ask) / 2)

        fair = self.get_fair_value(state, traderObject)
        take_width = 1
        clear_width = 0
        disregard_edge = 2
        join_edge = 1
        default_edge = 2
        soft_position_limit = 20

        buy_orders = []
        sell_orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            vol = -order_depth.sell_orders[best_ask]
            if best_ask <= fair - take_width and vol <= 15:
                qty = min(vol, self.LIMIT - position)
                if qty > 0:
                    buy_orders.append(Order(self.symbol, best_ask, qty))
                    buy_order_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair + take_width and vol <= 15:
                qty = min(vol, self.LIMIT + position)
                if qty > 0:
                    sell_orders.append(Order(self.symbol, best_bid, -qty))
                    sell_order_volume += qty

        net_pos = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair - clear_width)
        fair_ask = round(fair + clear_width)
        buy_qty = self.LIMIT - (position + buy_order_volume)
        sell_qty = self.LIMIT + (position - sell_order_volume)

        if net_pos > 0:
            clear_qty = sum(v for p, v in order_depth.buy_orders.items() if p >= fair_ask)
            sent_qty = min(sell_qty, min(clear_qty, net_pos))
            if sent_qty > 0:
                sell_orders.append(Order(self.symbol, fair_ask, -sent_qty))
                sell_order_volume += sent_qty

        if net_pos < 0:
            clear_qty = sum(-v for p, v in order_depth.sell_orders.items() if p <= fair_bid)
            sent_qty = min(buy_qty, min(clear_qty, -net_pos))
            if sent_qty > 0:
                buy_orders.append(Order(self.symbol, fair_bid, sent_qty))
                buy_order_volume += sent_qty

        asks_above_fair = [p for p in order_depth.sell_orders if p > fair + disregard_edge]
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair - disregard_edge]

        best_ask_above = min(asks_above_fair) if asks_above_fair else None
        best_bid_below = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair + default_edge)
        if best_ask_above:
            ask = best_ask_above if abs(best_ask_above - fair) <= join_edge else best_ask_above - 1

        bid = round(fair - default_edge)
        if best_bid_below:
            bid = best_bid_below if abs(fair - best_bid_below) <= join_edge else best_bid_below + 1

        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1

        if self.LIMIT - (position + buy_order_volume) > 0:
            buy_orders.append(Order(self.symbol, bid, self.LIMIT - (position + buy_order_volume)))
        if self.LIMIT + (position - sell_order_volume) > 0:
            sell_orders.append(Order(self.symbol, ask, -(self.LIMIT + (position - sell_order_volume))))

        self.orders.extend(buy_orders + sell_orders)

    def save(self):
        return {
            "ink_history": list(self.ink_history),
            "last_price": self.last_price
        }

    def load(self, data) -> None:
        if data:
            self.ink_history = deque(data.get("ink_history", []), maxlen=self.window_size)
            self.last_price = data.get("last_price", None)
            

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
        

class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
            # "CROISSANTS": Basket1ProductsStrategy("CROISSANTS", limits["CROISSANTS"]),
            "JAMS": Basket1ProductsStrategy("JAMS", limits["JAMS"]),
            "DJEMBES": Basket1ProductsStrategy("DJEMBES", limits["DJEMBES"]),
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", limits["PICNIC_BASKET2"]),
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", limits["PICNIC_BASKET1"])
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
