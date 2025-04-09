# from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Deque
from collections import deque
import json

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
                'window_size': 10,
                'max_position': 100,
                'price_history': {'asks': deque(maxlen=10), 'bids': deque(maxlen=10)},
                'last_trade_price': 0.0,
                'decay_factor': 0.2
            },
            'RAINFOREST_RESIN': {
                'window_size': 10,
                'max_position': 26,
                'price_history': {'asks': deque(maxlen=10), 'bids': deque(maxlen=10)},
                'last_trade_price': 0.0,
                'decay_factor': 0.2
            }
        }
        self.vwap_depth = 5

    def calculate_vwap(self, prices: List[float], volumes: List[int], fallback: float) -> float:
        """Safe VWAP calculation with boundary checks"""
        depth = min(self.vwap_depth, len(prices), len(volumes))
        if depth == 0 or sum(volumes[:depth]) < 10:
            return fallback
        return sum(p*v for p,v in zip(prices[:depth], volumes[:depth])) / sum(volumes[:depth])

    def run(self, state: TradingState):
        result = {}
        
        for product, order_depth in state.order_depths.items():
            if product not in self.product_params:
                continue
                
            params = self.product_params[product]
            orders = []
            current_position = state.position.get(product, 0)
            
            # Extract order book data
            bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            bid_volumes = [order_depth.buy_orders[p] for p in bid_prices]
            ask_prices = sorted(order_depth.sell_orders.keys())
            ask_volumes = [abs(order_depth.sell_orders[p]) for p in ask_prices]
            
            # Get best prices with fallbacks
            best_bid = bid_prices[0] if bid_prices else 0
            best_ask = ask_prices[0] if ask_prices else 0
            
            # Calculate valuation
            bid_vwap = self.calculate_vwap(bid_prices, bid_volumes, best_bid)
            ask_vwap = self.calculate_vwap(ask_prices, ask_volumes, best_ask)
            current_vwap = (bid_vwap + ask_vwap) / 2
            
            # Update price history for this product
            params['price_history']['asks'].append(best_ask)
            params['price_history']['bids'].append(best_bid)
            
            spread = best_ask - best_bid if best_ask and best_bid else 1.0
            
            # Product-specific trading logic
            target_volume = 0
            ask_history = params['price_history']['asks']
            bid_history = params['price_history']['bids']
            
            if len(ask_history) == params['window_size']:
                if best_ask <= min(ask_history):
                    if best_ask < current_vwap - spread/2:
                        max_buy = min(
                            sum(ask_volumes),
                            params['max_position'] - current_position
                        )
                        target_volume = max_buy
                        
            if len(bid_history) == params['window_size']:
                if best_bid >= max(bid_history):
                    if best_bid > current_vwap + spread/2:
                        max_sell = min(
                            sum(bid_volumes),
                            params['max_position'] + current_position
                        )
                        target_volume = -max_sell

            # Generate orders
            if target_volume > 0:  # Buy orders
                cumulative = 0
                for ask_price in ask_prices:
                    if cumulative >= target_volume:
                        break
                    if ask_price > current_vwap - spread/2:
                        continue
                    volume = min(abs(order_depth.sell_orders[ask_price]), target_volume - cumulative)
                    orders.append(Order(product, ask_price, volume))
                    cumulative += volume
                    
            elif target_volume < 0:  # Sell orders
                cumulative = 0
                for bid_price in bid_prices:
                    if cumulative >= abs(target_volume):
                        break
                    if bid_price < current_vwap + spread/2:
                        continue
                    volume = min(order_depth.buy_orders[bid_price], abs(target_volume) - cumulative)
                    orders.append(Order(product, bid_price, -volume))
                    cumulative += volume
            
            # Update last trade price for this product
            if orders:
                params['last_trade_price'] = orders[0].price
                
            result[product] = orders

        return result, 0, json.dumps({})