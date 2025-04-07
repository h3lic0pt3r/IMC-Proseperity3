# from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Deque
from collections import deque
import json
import numpy as np


class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol  # Product name
        self.price = price  # Price level
        self.quantity = quantity  # +ve = buy, -ve = sell

class OrderDepth:
    def __init__(self):
        self.buy_orders = {}
        self.sell_orders = {}

class TradingState:
    def __init__(self, timestamp, order_depths, position):
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.position = position

class Trader:
    def __init__(self):
        self.product_params = {
            'KELP': {
                'window_size': 12,
                'max_position': 80,
                'price_history': deque(maxlen=20),
                'adaptive_threshold': 0.5,  # Overly high threshold to ensure bad trades
                'spread_multiplier': 1.5,  # Aggressive spread usage for poor entries
            },
            'RAINFOREST_RESIN': {
                'window_size': 12,
                'max_position': 40,
                'price_history': deque(maxlen=20),
                'adaptive_threshold': 0.7,
                'spread_multiplier': 1.8,
            }
        }

    def run(self, state: TradingState):
        result = {}
        
        for product, order_depth in state.order_depths.items():
            if product not in self.product_params:
                continue
                
            params = self.product_params[product]
            orders = []
            current_position = state.position.get(product, 0)
            
            # Extract order book data
            bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            asks = sorted(order_depth.sell_orders.keys())
            
            if not bids or not asks:
                continue
                
            best_bid = bids[0]
            best_ask = asks[0]
            mid_price = (best_bid + best_ask) / 2
            
            # Update price history
            params['price_history'].append(mid_price)
            
            # Bad trading logic: Buy high and sell low
            if len(params['price_history']) >= params['window_size']:
                avg_price = sum(params['price_history']) / len(params['price_history'])
                
                # Buy when price is above average (high price)
                if mid_price > avg_price + params['adaptive_threshold']:
                    max_buy = min(
                        sum(abs(v) for v in order_depth.sell_orders.values()),
                        params['max_position'] - current_position
                    )
                    if max_buy > 0:
                        orders.append(Order(product, best_ask, -max_buy))
                
                # Sell when price is below average (low price)
                elif mid_price < avg_price - params['adaptive_threshold']:
                    max_sell = min(
                        sum(v for v in order_depth.buy_orders.values()),
                        params['max_position'] + current_position
                    )
                    if max_sell > 0:
                        orders.append(Order(product, best_bid, max_sell))
            
            result[product] = orders

        return result, 0, json.dumps({})
