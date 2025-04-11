from typing import Dict, List
from collections import deque
import json
import numpy as np
import statistics
from typing import Dict, List
from collections import deque
import json
import numpy as np
import statistics
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List
import math
import numpy as np
from collections import deque
import json
from typing import Any
class Order:
    def _init(self, symbol, price, quantity):  # fixed typo: _init → _init_
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class OrderDepth:
    def _init_(self):  # fixed typo
        self.buy_orders = {}
        self.sell_orders = {}

class TradingState:
    def _init_(self, timestamp, order_depths, position):  # fixed typo
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.position = position
class Logger:
    def _init_(self) -> None:
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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()
class Trader:
    def _init_(self):  # fixed typo
        self.product_params = {
        'KELP': {
            'strategy': 'fair_price_mm',
            'valuation_strategy': 'ema',  # 'true_value', 'vwap', 'mid', etc.
            'true_value': 2000.0,  # only used if valuation_strategy == 'true_value'
            'window_size': 3,
            'max_position': 50,
            'price_history': deque(maxlen=50),
            'ema': None,
            'buy_price': None
        },
        'RAINFOREST_RESIN': {
            'strategy': 'market_maker',
            'valuation_strategy': 'true_value',
            'true_value': 10000.0,
            'window_size': 3,
            'max_position': 50,
            'price_history': deque(maxlen=50),
            'ema': 10000,
            'base_spread': 3.5,
            'order_size': 50, 
            'skew_sensitivity': 0.02
        },
        'SQUID_INK': {
            'strategy': 'bollinger',
            'valuation_strategy': 'ema',  # Best bid + best ask / 2
            'true_value': 2000.0,
            'window_size': 3,
            'max_position': 50,
            'price_history': deque(maxlen=50),
            'ema': None
        }
    }
    def calculate_vwap(self, prices, volumes, fallback_price):
        total_volume = sum(volumes)
        if total_volume == 0:
            return fallback_price  # fallback in case there's no volume
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return vwap


    def get_mid_price(self, product, order_depth):
        params = self.product_params[product]
        strategy = params.get('valuation_strategy', 'ema')

        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        best_bid = max(bids) if bids else 0
        best_ask = min(asks) if asks else 0

        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        if strategy == 'true_value':
            return params.get('true_value', mid_price)

        elif strategy == 'mid':
            return mid_price

        elif strategy == 'vwap':
            bid_prices = sorted(bids.keys(), reverse=True)
            bid_volumes = [bids[p] for p in bid_prices]
            ask_prices = sorted(asks.keys())
            ask_volumes = [abs(asks[p]) for p in ask_prices]

            vwap_bid = self.calculate_vwap(bid_prices, bid_volumes, best_bid)
            vwap_ask = self.calculate_vwap(ask_prices, ask_volumes, best_ask)
            return (vwap_bid + vwap_ask) / 2

        elif strategy == 'ema':
            alpha = 2 / (params['window_size'] + 1)
            if params.get('ema') is None:
                params['ema'] = mid_price
            else:
                params['ema'] = alpha * mid_price + (1 - alpha) * params['ema']
            return params['ema']

        return mid_price  # fallback
    # Add this in your Trader class
    def market_maker_strategy(self, product, mid_price, state, order_depth):
        p = self.product_params[product]
        orders = []

        price_history = p['price_history']
        price_history.append(mid_price)
        if len(price_history) < 5:
            return []

        position = state.position.get(product, 0)
        max_position = p.get('max_position', 50)

        # === Dynamic Spread based on volatility ===
        if len(p['price_history']) < 10:
            recent_returns=[0,0,0,0,0,0,0,0,0,0,0]
        else:
            recent_returns = [price_history[-i] - price_history[-i - 1] for i in range(1, 10)]
        volatility = max(1, statistics.stdev(recent_returns))  # avoid zero
        base_spread = p.get('base_spread', 2)
        spread = base_spread + 0.05 * volatility  # wider in volatility

        # === Dynamic Skew based on inventory and trend ===
        skew_sensitivity = p.get('skew_sensitivity', 0.1)
        price_trend = sum(recent_returns[-4:])
        skew = skew_sensitivity * position - 0.2 * price_trend

        # === Fixed Order Size ===
        order_size = p.get('order_size', 5)

        # Calculate bid/ask prices
        bid_price = int(mid_price - spread / 2 - skew)
        ask_price = int(mid_price + spread / 2 - skew)

        # Limit quantity to not exceed max position
        bid_qty = min(order_size, max_position - position)
        ask_qty = min(order_size, max_position + position)

        if bid_qty > 0:
            #logger.print(f"[{product}] MM Buy {bid_qty} @ {bid_price} (skew: {skew:.2f}, spread: {spread:.2f})")
            orders.append(Order(product, bid_price, bid_qty))

        if ask_qty > 0:
            #logger.print(f"[{product}] MM Sell {ask_qty} @ {ask_price} (skew: {skew:.2f}, spread: {spread:.2f})")
            orders.append(Order(product, ask_price, -ask_qty))

        return orders
    # def market_maker_strategy(self, product, mid_price, state, order_depth):
    #     p = self.product_params[product]
    #     orders = []

    #     # Basic config
    #     spread = p.get('spread', 2)
    #     order_size = p.get('order_size', 5)
    #     max_position = p.get('max_position', 50)
    #     position = state.position.get(product, 0)

    #     # Skew factor: how much to shift bid/ask prices based on inventory
    #     skew_sensitivity = p.get('skew_sensitivity', 0.1)
    #     skew = skew_sensitivity * position

    #     # Apply skew to bid and ask prices
    #     bid_price = int(mid_price - spread / 2 - skew)
    #     ask_price = int(mid_price + spread / 2 - skew)

    #     # Adjust order sizes to avoid exceeding max position
    #     bid_qty = min(order_size, max_position - position)
    #     ask_qty = min(order_size, max_position + position)

    #     if bid_qty > 0:
    #         #logger.print(f"[{product}] MM Buy {bid_qty} @ {bid_price} (skew: {skew})")
    #         orders.append(Order(product, bid_price, bid_qty))

    #     if ask_qty > 0:
    #         #logger.print(f"[{product}] MM Sell {ask_qty} @ {ask_price} (skew: {skew})")
    #         orders.append(Order(product, ask_price, -ask_qty))

    def bollinger_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < p['window_size']:
            return []

        prices = list(p['price_history'])
        mean = np.mean(prices)
        std = np.std(prices)
        upper = mean + 2 * std
        lower = mean - 2 * std

        #logger.print(f"[{product}] Bollinger Bands: mean={mean:.2f}, upper={upper:.2f}, lower={lower:.2f}")

        orders = []
        current_position = state.position.get(product, 0)

        if mid_price < lower:
            qty = min(10, p['max_position'] - current_position)
            #logger.print(f"[{product}] Bollinger Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))

        elif mid_price > upper:
            qty = min(10, p['max_position'] + current_position)
            #logger.print(f"[{product}] Bollinger Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def breakout_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < p['window_size']:
            return []

        prices = list(p['price_history'])[:-1]
        high = max(prices)
        low = min(prices)

        #logger.print(f"[{product}] Breakout: high={high:.2f}, low={low:.2f}, current={mid_price:.2f}")

        orders = []
        current_position = state.position.get(product, 0)

        if mid_price > high:
            qty = min(10, p['max_position'] - current_position)
            #logger.print(f"[{product}] Breakout Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))
        elif mid_price < low:
            qty = min(10, p['max_position'] + current_position)
            #logger.print(f"[{product}] Breakout Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def moving_average_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < p['window_size']:
            return []

        avg = np.mean(p['price_history'])

        #logger.print(f"[{product}] Moving Average: mean={avg:.2f}, current={mid_price:.2f}")

        orders = []
        current_position = state.position.get(product, 0)

        if mid_price > avg:
            qty = min(10, p['max_position'] - current_position)
            #logger.print(f"[{product}] MA Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))
        elif mid_price < avg:
            qty = min(10, p['max_position'] + current_position)
            #logger.print(f"[{product}] MA Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def zscore_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < p['window_size']:
            return []

        mean = np.mean(p['price_history'])
        std = np.std(p['price_history'])
        z = (mid_price - mean) / std if std else 0
        #logger.print(f"[{product}] Z-Score: {z:.2f}")

        orders = []
        current_position = state.position.get(product, 0)

        if z < 1:
            qty = min(25, p['max_position'] - current_position)
            #logger.print(f"[{product}] Z-Score Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))
        elif z > 1:
            qty = min(25, p['max_position'] + current_position)
            #logger.print(f"[{product}] Z-Score Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def crossover_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < 7:
            return []

        short = np.mean(list(p['price_history'])[-3:])
        long = np.mean(list(p['price_history'])[-7:])
        #logger.print(f"[{product}] Crossover: short={short:.2f}, long={long:.2f}")

        orders = []
        current_position = state.position.get(product, 0)

        if short > long:
            qty = min(10, p['max_position'] - current_position)
            #logger.print(f"[{product}] Crossover Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))
        elif short < long:
            qty = min(10, p['max_position'] + current_position)
            #logger.print(f"[{product}] Crossover Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def momentum_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < 4:
            return []

        changes = [p['price_history'][i] - p['price_history'][i - 1] for i in range(1, len(p['price_history']))]
        #logger.print(f"[{product}] Momentum changes: {changes[-4:]}")

        orders = []
        current_position = state.position.get(product, 0)

        if changes[-1] > 0 and changes[-2] > 0:
            qty = min(10, p['max_position'] - current_position)
            p['buy_price'] = mid_price
            #logger.print(f"[{product}] Momentum Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))
        elif all(c < 0 for c in changes[-3:]) or (p['buy_price'] and mid_price < 0.8 * p['buy_price']):
            qty = min(10, p['max_position'] + current_position)
            p['buy_price'] = None
            #logger.print(f"[{product}] Momentum Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))

        return orders

    def fair_price_mm_strategy(self, product, order_depth, state):
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)
        if best_bid == 0 or best_ask == 0:
            return []

        fair_price = (best_bid + best_ask) / 2
        #logger.print(f"[{product}] Fair Price MM: best_bid={best_bid}, best_ask={best_ask}, fair_price={fair_price}")

        orders = []
        current_position = state.position.get(product, 0)
        max_position = self.product_params[product]['max_position']

        buy_qty = min(10, max_position - current_position)
        sell_qty = min(10, max_position + current_position)

        orders.append(Order(product, int(fair_price - 1), buy_qty))
        orders.append(Order(product, int(fair_price + 1), -sell_qty))

        #logger.print(f"[{product}] Market Making Buy {buy_qty} at {int(fair_price - 1)}")
        #logger.print(f"[{product}] Market Making Sell {sell_qty} at {int(fair_price + 1)}")
        return orders

    def trend_follow_sl_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < 3:
            return []

        changes = [p['price_history'][-i] - p['price_history'][-i - 1] for i in range(1, 3)]
        #logger.print(f"[{product}] Trend SL changes: {changes[::-1]}")

        orders = []
        current_position = state.position.get(product, 0)

        if changes[-1] > 0 and changes[-2] > 0:
            qty = min(10, p['max_position'] - current_position)
            p['buy_price'] = mid_price
            #logger.print(f"[{product}] Trend Buy {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), qty))

        elif p.get('buy_price') and mid_price < 0.8 * p['buy_price']:
            qty = min(10, p['max_position'] + current_position)
            #logger.print(f"[{product}] Trend Stop Loss Sell {qty} at {mid_price}")
            orders.append(Order(product, int(mid_price), -qty))
            p['buy_price'] = None

        return orders
    def orderbook_imbalance_strategy(self, product, order_depth, state):
        orders = []
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        best_bid = max(bids.keys(), default=0)
        best_ask = min(asks.keys(), default=0)
        bid_volume = sum(bids.values())
        ask_volume = sum(abs(v) for v in asks.values())
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume != 0 else 0
        ##logger.print(f"[{product}] Orderbook Imbalance: {imbalance:.2f}")

        current_position = state.position.get(product, 0)
        max_position = self.product_params[product]['max_position']

        if imbalance > 0.3:
            volume = min(max_position - current_position, 10)
            orders.append(Order(product, best_ask, volume))
            ##logger.print(f"[{product}] Buying {volume} at {best_ask} due to OB imbalance")
        elif imbalance < -0.3:
            volume = min(max_position + current_position, 10)
            orders.append(Order(product, best_bid, -volume))
            ##logger.print(f"[{product}] Selling {volume} at {best_bid} due to OB imbalance")

        return orders

    def keltner_channel_strategy(self, product, mid_price, state):
        p = self.product_params[product]
        p['price_history'].append(mid_price)
        if len(p['price_history']) < 10:
            return []

        ema = sum(p['price_history']) / len(p['price_history'])
        atr = sum(abs(p['price_history'][i] - p['price_history'][i - 1]) for i in range(1, len(p['price_history']))) / (len(p['price_history']) - 1)
        upper_band = ema + 1.5 * atr
        lower_band = ema - 1.5 * atr

        ##logger.print(f"[{product}] Keltner Channel: EMA={ema:.2f}, ATR={atr:.2f}, Upper={upper_band:.2f}, Lower={lower_band:.2f}")

        orders = []
        current_position = state.position.get(product, 0)
        max_position = p['max_position']

        if mid_price < lower_band:
            qty = min(10, max_position - current_position)
            orders.append(Order(product, int(mid_price), qty))
            ##logger.print(f"[{product}] Buy {qty} at {mid_price} (Below Keltner Lower Band)")
        elif mid_price > upper_band:
            qty = min(10, max_position + current_position)
            orders.append(Order(product, int(mid_price), -qty))
            ##logger.print(f"[{product}] Sell {qty} at {mid_price} (Above Keltner Upper Band)")

        return orders


    def run(self, state: TradingState):
        result = {}
        for product, order_depth in state.order_depths.items():
            if product not in self.product_params:
                continue

            strategy = self.product_params[product]['strategy']
            mid_price = self.get_mid_price(product, order_depth)
            #logger.print(f"\n=== {product} @ {mid_price:.2f} using {strategy} strategy ===")

            if strategy == 'zscore':
                result[product] = self.zscore_strategy(product, mid_price, state)
            elif strategy == 'crossover':
                result[product] = self.crossover_strategy(product, mid_price, state)
            elif strategy == 'momentum':
                result[product] = self.momentum_strategy(product, mid_price, state)
            elif strategy == 'bollinger':
                result[product] = self.bollinger_strategy(product, mid_price, state)
            elif strategy == 'breakout':
                result[product] = self.breakout_strategy(product, mid_price, state)
            elif strategy == 'moving_average':
                result[product] = self.moving_average_strategy(product, mid_price, state)
            elif strategy == 'fair_price_mm':
                result[product] = self.fair_price_mm_strategy(product, order_depth, state)
            elif strategy == 'trend_follow_sl':
                result[product] = self.trend_follow_sl_strategy(product, mid_price, state)
            elif strategy == 'orderbook_imbalance':
                result[product] = self.orderbook_imbalance_strategy(product, order_depth, state)
            elif strategy == 'keltner_channel':
                result[product] = self.keltner_channel_strategy(product, mid_price, state)
            elif strategy == 'stable_mm':
                result[product] = self.crossover_strategy(product, mid_price, state)
            elif strategy == 'market_maker':
                result[product] = self.market_maker_strategy(product, mid_price, state, order_depth)
            else:
                result[product] = []


        logger.flush(state, result, 0, "")
        return result, 0, json.dumps({})