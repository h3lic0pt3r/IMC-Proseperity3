
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List
import math
import numpy as np
from collections import deque
import json
from typing import Any


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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.T = 100000 #Trading Time
        self.product_data={
            'KELP': {
                'sigma': 2026 * 0.02 / math.sqrt(self.T),
                'max_position': 100,
                'k': math.log(2) / 0.01,
                'gamma' : 0.01/100,
                'price_history': deque(maxlen=10)
            },
            'RAINFOREST_RESIN': {
                'sigma' : 10000 * 0.02 / math.sqrt(self.T),
                'max_position': 26,
                'k': math.log(2) / 0.01,
                'gamma' : 0.01/26,
                'price_history': deque(maxlen=10)
            },            
            'SQUID_INK': {
                'sigma' : 2000 * 0.02 / math.sqrt(self.T),
                'max_position': 100,
                'k': math.log(2) / 0.01,
                'gamma' : 0.01/100,
                'price_history': deque(maxlen=10)
            }
        }

    def run(self, state: TradingState):
        result = {}
        for product, order_depth in state.order_depths.items():
            if product not in self.product_data:
                continue
                
            params = self.product_data[product]
            orders = []
            current_position = state.position.get(product, 0)
            
            # Get order book prices
            bids = sorted(order_depth.buy_orders.keys(), reverse=True) or [0]
            asks = sorted(order_depth.sell_orders.keys()) or [float('inf')]
            best_bid = bids[0]
            best_ask = asks[0]
            mid_price = (best_bid + best_ask) / 2

            # CORRECTED FORMULA (uses T - timestamp instead of normalized time)
            time_left = self.T - state.timestamp
            spread = (params['gamma'] * (params['sigma']**2) * time_left) + \
                    (2/params['gamma']) * math.log(1 + params['gamma']/params['k'])
            
            rest_price = mid_price - current_position * params['gamma'] * (params['sigma']**2) * time_left
            
            bid_price = int(rest_price - spread/2)
            ask_price = int(rest_price + spread/2)

            # 1. Take existing liquidity
            for ask, vol in sorted(order_depth.sell_orders.items()):
                if ask <= bid_price:
                    max_buy = min(params['max_position'] - current_position, -vol)
                    if max_buy > 0:
                        orders.append(Order(product, ask, max_buy))
                        current_position += max_buy

            for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid >= ask_price:
                    max_sell = min(params['max_position'] + current_position, vol)
                    if max_sell > 0:
                        orders.append(Order(product, bid, -max_sell))
                        current_position -= max_sell

            # 2. Add new limit orders if no matches
            if not orders:
                # Place bids 1 cent below theoretical price
                bid_price = min(bid_price, best_bid - 1)
                ask_price = max(ask_price, best_ask + 1)
                
                buy_qty = params['max_position'] - current_position
                sell_qty = params['max_position'] + current_position
                
                if buy_qty > 0:
                    orders.append(Order(product, bid_price, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, ask_price, -sell_qty))

            result[product] = orders
            logger.print("-----------------------",bid_price, ask_price, rest_price,"------------------------")
        logger.flush(state, result, 0, "")
        return result, 0, ""
