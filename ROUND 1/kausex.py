from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
from collections import deque

class Trader:
    def __init__(self):
        self.max_position = 20  # Reduced for safety
        self.price_history = deque(maxlen=10)
        self.spread = 1.0  # Minimum spread between bid/ask
        self.position = {}

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """Calculate fair value using best bid/ask prices with fallbacks"""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask else 0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            if product not in self.position:
                self.position[product] = 0
            
            current_position = state.position.get(product, 0)
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            
            # Update price history
            if best_bid and best_ask:
                self.price_history.append((best_bid + best_ask) / 2)
            
            # Calculate simple moving average
            moving_avg = sum(self.price_history)/len(self.price_history) if self.price_history else 0
            
            orders = []
            
            # Buy logic
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                if best_ask < moving_avg and current_position < self.max_position:
                    volume = min(abs(order_depth.sell_orders[best_ask]), 
                               self.max_position - current_position)
                    if volume > 0:
                        orders.append(Order(product, best_ask, volume))
            
            # Sell logic
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                if best_bid > moving_avg and current_position > -self.max_position:
                    volume = min(order_depth.buy_orders[best_bid], 
                               self.max_position + current_position)
                    if volume > 0:
                        orders.append(Order(product, best_bid, -volume))
            
            result[product] = orders

        return result, conversions, trader_data
