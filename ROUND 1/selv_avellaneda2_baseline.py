
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
                'sigma': 2034 * 0.02 / math.sqrt(self.T),
                'max_position': 50,
                'k': 8,
                'gamma' : 1e-9,
                'price_history': deque(maxlen=10),
                'prev_mid' : 2034,               # Smoothed mid-price
                'alpha' : 0.2,                 # EWMA smoothing
                'deviation_threshold' : 35,    # Deviation to flag fake
                'cooldown' : 0,            # Cooldown after fake detection
                'cooldown_period' : 5,         # Don't quote for these many ticks

            },
            'RAINFOREST_RESIN': {
                'sigma' : 10000 * 0.02 / math.sqrt(self.T),
                'max_position': 50,
                'k': 5,     ## smaller implies more volatile market
                'gamma' : 1e-9,  ##smaller implies more agressive betting
                'price_history': deque(maxlen=10),
                'prev_mid' : 10000,               # Smoothed mid-price
                'alpha' : 0.2,                 # EWMA smoothing
                'deviation_threshold' : 45,    # Deviation to flag fake
                'cooldown' : 0,            # Cooldown after fake detection
                'cooldown_period' : 5,         # Don't quote for these many ticks
            },            
            'SQUID_INK': {
                'sigma' : 1834 * 0.02 / math.sqrt(self.T),
                'max_position': 50,
                'k': 5,
                'gamma' : 1e-9,
                'price_history': deque(maxlen=10),
                'prev_mid' : 1834,               # Smoothed mid-price
                'alpha' : 0.2,                 # EWMA smoothing
                'deviation_threshold' : 30,    # Deviation to flag fake
                'cooldown' : 0,            # Cooldown after fake detection
                'cooldown_period' : 5,         # Don't quote for these many ticks
            }
        }

    def calculate_volatility(self, price_history: deque) -> float:              
        """Calculates annualized volatility from price history using log returns."""
        if len(price_history) < 2:
            return 0  # Not enough data
        
        returns = []
        prev_price = price_history[0]
        for price in list(price_history)[1:]:
            returns.append(math.log(price / prev_price))
            prev_price = price
            
        if len(returns) < 1:
            return 0
            
        std_dev = np.std(returns)
        annualized_vol = std_dev 
        return annualized_vol

    def calculate_k(self, order_depth: OrderDepth, mid_price: float) -> float:
        """Estimate the liquidity parameter k using exponential decay model on all available order book levels."""
        
        def estimate_k_from_side(levels, side: str) -> float | None:
            deltas = []
            log_volumes = []
            for price, volume in levels:
                if side == "bid" and volume <= 0:
                    continue
                if side == "ask" and volume >= 0:
                    continue
                vol = abs(volume)
                delta = abs(price - mid_price)
                if vol > 0 and delta > 0:
                    deltas.append(delta)
                    log_volumes.append(math.log(vol))
            
            if len(deltas) >= 2:
                # Fit log(volume) = log(V0) - k * delta  -->  y = -k * x + b
                A = np.vstack([deltas, np.ones(len(deltas))]).T
                slope, _ = np.linalg.lstsq(A, -np.array(log_volumes), rcond=None)[0]
                if 0.01 <= slope <= 10:  # Optional sanity check
                    return slope
            return None

        bids = sorted(order_depth.buy_orders.items(), reverse=True)[:5]
        asks = sorted(order_depth.sell_orders.items())[:5]

        k_values = []
        k_bid = estimate_k_from_side(bids, "bid")
        if k_bid is not None:
            k_values.append(k_bid)

        k_ask = estimate_k_from_side(asks, "ask")
        if k_ask is not None:
            k_values.append(k_ask)

        return float(np.mean(k_values)) if k_values else 1.5  # Default fallback

    def detect_fake(self, product, mid_price): #detects fake drops or spikes and bets against the trend
        params = self.product_data[product]
        smoothed = params['prev_mid']
        ewma_mid = params['alpha'] * mid_price + (1 - params['alpha']) * smoothed
        params['prev_mid'] = ewma_mid
        
        deviation = abs(mid_price - ewma_mid)

        # Cooldown active? Don't quote
        if params['cooldown'] > 0:
            params['cooldown'] -= 1
            return "cooldown", ewma_mid

        # Detect large deviation without trade confirmation (simplified condition)
        if deviation > params['deviation_threshold'] :
            params['cooldown'] = params['cooldown_period']
            return "fake", ewma_mid

        return "normal", ewma_mid


    def run(self, state: TradingState):
        result = {}
        for product, order_depth in state.order_depths.items():
            if product not in self.product_data:
                continue
                
            params = self.product_data[product]
            orders = []
            current_position = state.position.get(product, 0)
            
            # Get order book prices
            if order_depth.buy_orders:
                # Find bid with maximum volume
                max_bid_volume = max(order_depth.buy_orders.values())
                candidate_bids = [price for price, vol in order_depth.buy_orders.items() if vol == max_bid_volume]
                best_bid = max(candidate_bids)  # Highest price among max volume bids
            else:
                best_bid = 0

            if order_depth.sell_orders:
                # Find ask with maximum volume (using absolute value)
                max_ask_volume = max(abs(vol) for vol in order_depth.sell_orders.values())
                candidate_asks = [price for price, vol in order_depth.sell_orders.items() if abs(vol) == max_ask_volume]
                best_ask = min(candidate_asks)  # Lowest price among max volume asks
            else:
                best_ask = float('inf')

            # Only proceed if we have valid prices
            if best_bid == 0 or best_ask == float('inf'):
                continue
                
            mid_price = (best_bid + best_ask) / 2

            params['price_history'].append(mid_price)
            realized_vol = self.calculate_volatility(params['price_history'])
            effective_sigma = realized_vol if realized_vol > 0 else params['sigma']
            
            params['k'] = self.calculate_k(order_depth, mid_price)
            gamma = max(params['gamma']/(1+ 20 * realized_vol), 1e-10)

            # Spread calculation with dynamic volatility
            time_left = (self.T - state.timestamp)/self.T
            spread = (gamma * (effective_sigma**2) * time_left) + \
                    (2/gamma) * math.log(1 + gamma/params['k'])            
            market_spread = best_ask - best_bid

            rest_price = mid_price - current_position * gamma * (effective_sigma**2) * time_left
            
            bid_price = int(rest_price - spread/2)
            ask_price = int(rest_price + spread/2)

            # inventory_ratio = abs(current_position) / params['max_position']
            # aggression_factor = 1 + 3.5*inventory_ratio  # ranges from 1 to 3

            # 2. Add new limit orders if no matches
            num_levels = 3  # Number of price levels
            level_spacing = min(spread , market_spread)/(2*num_levels)
            
            # Clear existing positions if needed
            remaining_buy = params['max_position'] - current_position
            remaining_sell = params['max_position'] + current_position

            status, ewma_mid = self.detect_fake(product, mid_price)

            if status == "cooldown":
                # logger.print(f"{product} in cooldown — not quoting")
                continue  # Don't quote during cooldown

            elif status == "fake":
                # logger.print(f"{product} FAKE detected at {mid_price}, smoothed {ewma_mid}")
                # Reversion bet: fade the move
                # position = state.position.get(product, 0)
                
                if mid_price < ewma_mid:
                    # Price spiked down — BUY
                    orders.append(Order(product, int(mid_price), remaining_buy))
                else:
                    # Price spiked up — SELL
                    orders.append(Order(product, int(mid_price), -remaining_sell))
                result[product] = orders
                continue  # Skip the Avellaneda-Stoikov quoting


            # 1. Take existing liquidity
            for ask, vol in sorted(order_depth.sell_orders.items()):
                if ask <= bid_price:
                    max_buy = min(params['max_position'] - current_position, -vol)
                    if max_buy > 0:
                        orders.append(Order(product, ask, max_buy))
                        remaining_buy -= max_buy
                        current_position += max_buy

            for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid >= ask_price:
                    max_sell = min(params['max_position'] + current_position, vol)
                    if max_sell > 0:
                        orders.append(Order(product, bid, -max_sell))
                        remaining_sell -=max_sell
                        current_position -= max_sell


            if remaining_buy > 0 or remaining_sell > 0:
                for i in range(num_levels):
                    # Calculate level prices
                    bid_level_price = int(rest_price - (i + 1) * level_spacing)
                    ask_level_price = int(rest_price + (i + 1) * level_spacing)
                    
                    # Calculate size for each level (decreasing with distance)
                    level_factor = (num_levels - i) / num_levels
                    bid_size = int(remaining_buy * level_factor / num_levels)
                    ask_size = int(remaining_sell * level_factor / num_levels)
                    
                    # Place orders if size > 0
                    if bid_size > 0:
                        orders.append(Order(product, bid_level_price, bid_size))
                    if ask_size > 0:
                        orders.append(Order(product, ask_level_price, -ask_size))

            result[product] = orders
            logger.print("-----------------------",bid_price, ask_price, rest_price,params['k'],"------------------------")
        logger.flush(state, result, 0, "")
        return result, 0, ""
