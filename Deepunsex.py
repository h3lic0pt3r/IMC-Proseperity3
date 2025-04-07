from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import json

class Trader:
    DECAY_FACTOR = 0.2
    MAX_TRADE_SIZE = 10
    BOLLINGER_BANDS_MULTIPLIER = 0.5  # Reduced multiplier for more frequent trading

    def run(self, state: TradingState):
        result = {}
        
        # Load previous EMA and variance data
        if state.traderData:
            product_data = json.loads(state.traderData)
        else:
            product_data = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                current_mid = (best_bid + best_ask) / 2

                # Initialize or update EMA/variance
                if product not in product_data:
                    product_data[product] = {
                        'ema': current_mid,
                        'variance': 0.0,
                        'last_price': current_mid
                    }
                else:
                    prev_ema = product_data[product]['ema']
                    prev_var = product_data[product]['variance']
                    prev_price = product_data[product]['last_price']
                    
                    # Update EMA with decay factor
                    new_ema = self.DECAY_FACTOR * current_mid + (1 - self.DECAY_FACTOR) * prev_ema
                    
                    # Update variance
                    error = current_mid - prev_ema
                    new_var = self.DECAY_FACTOR * (error**2) + (1 - self.DECAY_FACTOR) * prev_var
                    
                    product_data[product].update(ema=new_ema, variance=new_var, last_price=current_mid)

                ema = product_data[product]['ema']
                std_dev = np.sqrt(product_data[product]['variance']) if product_data[product]['variance'] > 0 else 1e-6
                
                # Calculate Bollinger Bands
                upper_band = ema + self.BOLLINGER_BANDS_MULTIPLIER * std_dev
                lower_band = ema - self.BOLLINGER_BANDS_MULTIPLIER * std_dev

                # Process sell orders (buy opportunities)
                for ask_price, ask_quantity in order_depth.sell_orders.items():
                    if ask_price < lower_band:
                        trade_size = min(abs(ask_quantity), self.MAX_TRADE_SIZE)
                        if trade_size > 0:
                            orders.append(Order(product, ask_price, trade_size))

                # Process buy orders (sell opportunities)
                for bid_price, bid_quantity in order_depth.buy_orders.items():
                    if bid_price > upper_band:
                        trade_size = min(abs(bid_quantity), self.MAX_TRADE_SIZE)
                        if trade_size > 0:
                            orders.append(Order(product, bid_price, -trade_size))

                result[product] = orders

        # Save updated state
        traderData = json.dumps(product_data)
        conversions = sum(len(orders) for orders in result.values())
        return result, conversions, traderData
