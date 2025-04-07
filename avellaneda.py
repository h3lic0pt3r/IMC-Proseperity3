from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Deque
from collections import deque
import json
import numpy as np

class Trader:
    def __init__(self):
        self.product_params = {
            'KELP': {
                'window_size': 15,
                'max_position': 60,
                'price_history': deque(maxlen=30),
                'entry_threshold': 0.8,
                'exit_threshold': 1.2,
                'stop_loss': 2.0,
                'take_profit': 1.5
            },
            'RAINFOREST_RESIN': {
                'window_size': 15,
                'max_position': 30,
                'price_history': deque(maxlen=30),
                'entry_threshold': 0.7,
                'exit_threshold': 1.3,
                'stop_loss': 2.2,
                'take_profit': 1.8
            }
        }

    def calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices)/prices[:-1]
        return np.std(returns)

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
            bid_volumes = [order_depth.buy_orders[p] for p in bids]
            asks = sorted(order_depth.sell_orders.keys())
            ask_volumes = [abs(order_depth.sell_orders[p]) for p in asks]
            
            # Get best prices
            best_bid = bids[0] if bids else 0
            best_ask = asks[0] if asks else 0
            mid_price = (best_bid + best_ask) / 2
            
            # Update price history
            params['price_history'].append(mid_price)
            price_history = list(params['price_history'])  # Convert deque to list
            
            # Calculate momentum safely
            momentum = 0.0
            if len(price_history) >= 20:
                momentum = np.mean(price_history[-5:]) - np.mean(price_history[-20:])
                
            # Rest of your trading logic remains unchanged
            # ... (VWAP calculation, order generation, etc)
            
            result[product] = orders

        return result, 0, json.dumps({})
