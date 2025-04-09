from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Deque
from collections import deque
import numpy as np
import json

class Trader:
    def __init__(self):
        self.product_params = {
            'KELP': {
                'window_size': 20,
                'max_position': 200,  # Increased position limit
                'price_history': deque(maxlen=50),
                'entry_threshold': 0.15,  # More aggressive entry
                'take_profit': 2.5,      # Higher profit target
                'stop_loss': 1.8,        # Tighter risk control
                'vwap_depth': 10         # Deeper order book analysis
            },
            'RAINFOREST_RESIN': {
                'window_size': 25,
                'max_position': 150,
                'price_history': deque(maxlen=60),
                'entry_threshold': 0.2,
                'take_profit': 3.0,
                'stop_loss': 2.2,
                'vwap_depth': 10
            }
        }

    def calculate_vwap(self, prices: List[int], volumes: List[int]) -> float:
        total_volume = sum(volumes[:self.product_params['KELP']['vwap_depth']])
        if total_volume == 0:
            return 0.0
        return sum(p*v for p,v in zip(prices, volumes)) / total_volume

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
            
            # Update price history as list
            params['price_history'].append(mid_price)
            price_list = list(params['price_history'])
            
            # Calculate momentum safely
            momentum = 0.0
            if len(price_list) >= 20:
                momentum = np.mean(price_list[-5:]) - np.mean(price_list[-20:])
                
            # Calculate VWAP
            bid_vwap = self.calculate_vwap(bids, bid_volumes)
            ask_vwap = self.calculate_vwap(asks, ask_volumes)
            current_vwap = (bid_vwap + ask_vwap) / 2
            
            # Position-based scaling
            position_ratio = abs(current_position)/params['max_position']
            size_multiplier = 1 - position_ratio**2  # Quadratic decay
            
            # Trading logic
            target_volume = 0
            spread = best_ask - best_bid
            
            if momentum > params['entry_threshold']:
                buy_price = current_vwap - spread * 0.25
                max_buy = min(
                    sum(ask_volumes),
                    int((params['max_position'] - current_position) * size_multiplier)
                )
                target_volume = max_buy
                
            elif momentum < -params['entry_threshold']:
                sell_price = current_vwap + spread * 0.3
                max_sell = min(
                    sum(bid_volumes),
                    int((params['max_position'] + current_position) * size_multiplier)
                )
                target_volume = -max_sell

            # Order execution
            if target_volume > 0:
                cumulative = 0
                for ask_price in asks:
                    if ask_price > current_vwap or cumulative >= target_volume:
                        break
                    volume = min(abs(order_depth.sell_orders[ask_price]), target_volume - cumulative)
                    orders.append(Order(product, ask_price, volume))
                    cumulative += volume
                    
            elif target_volume < 0:
                cumulative = 0
                for bid_price in bids:
                    if bid_price < current_vwap or cumulative >= abs(target_volume):
                        break
                    volume = min(order_depth.buy_orders[bid_price], abs(target_volume) - cumulative)
                    orders.append(Order(product, bid_price, -volume))
                    cumulative += volume
            
            # Profit taking and stop loss
            if current_position != 0:
                avg_cost = np.mean([o.price for o in orders]) if orders else mid_price
                
                # Trailing stop loss
                if current_position > 0 and best_bid < avg_cost * (1 - params['stop_loss']/100):
                    orders.append(Order(product, best_bid, -current_position))
                elif current_position < 0 and best_ask > avg_cost * (1 + params['stop_loss']/100):
                    orders.append(Order(product, best_ask, -current_position))
                
                # Partial profit taking
                if current_position > 0 and best_ask > avg_cost * (1 + params['take_profit']/100):
                    close_qty = int(current_position * 0.5)
                    orders.append(Order(product, best_ask, -close_qty))
                elif current_position < 0 and best_bid < avg_cost * (1 - params['take_profit']/100):
                    close_qty = int(abs(current_position) * 0.5)
                    orders.append(Order(product, best_bid, close_qty))
            
            result[product] = orders

        return result, 0, json.dumps({})
