from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List
import math
import numpy as np
from collections import deque, defaultdict
import json
from typing import Any
import statistics
from math import log, sqrt, exp, pi, erf
from typing import Literal, Tuple


##############      ALL IMPORTS     ##############


##############      LOGGER STUFF    ############## 



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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()
#########   END OF LOGGER ########


#########  PRODUCT PARAMS ########
max_position = {
    'KELP' : 50, 
    'RAINFOREST_RESIN' : 50, 
    'SQUID_INK' : 50, 
    'JAMS' : 350, 
    'CROISSANTS' : 250, 
    'DJEMBES' : 60, 
    'PICNIC_BASKET1' : 60, 
    'PICNIC_BASKET2' : 100,
    'VOLCANIC_ROCK': 400,
    'VOLCANIC_ROCK_VOUCHER_10000': 200,
    'VOLCANIC_ROCK_VOUCHER_10250': 200,
    'VOLCANIC_ROCK_VOUCHER_10500': 200,
    'VOLCANIC_ROCK_VOUCHER_9500': 200,
    'VOLCANIC_ROCK_VOUCHER_9750': 200
}
valuation_strategy={
    'KELP' : 'mid', 
    'RAINFOREST_RESIN' : 'mid', 
    'SQUID_INK' : 'mid', 
    'JAMS' : 'mid', 
    'CROISSANTS' : 'mid', 
    'DJEMBES' : 'mid', 
    'PICNIC_BASKET1' : 'mid', 
    'PICNIC_BASKET2' : 'mid',
    'VOLCANIC_ROCK': 'mid',
    'VOLCANIC_ROCK_VOUCHER_10000': 'mid',
    'VOLCANIC_ROCK_VOUCHER_10250': 'mid',
    'VOLCANIC_ROCK_VOUCHER_10500': 'mid',
    'VOLCANIC_ROCK_VOUCHER_9500': 'mid',
    'VOLCANIC_ROCK_VOUCHER_9750': 'mid' 
}


price_history = defaultdict(lambda: deque(maxlen=20))
cooldown = {
    'KELP' : 0, 
    'RAINFOREST_RESIN' : 0, 
    'SQUID_INK' : 0, 
    'JAMS' : 0, 
    'CROISSANTS' : 0, 
    'DJEMBES' : 0, 
    'PICNIC_BASKET1' : 0, 
    'PICNIC_BASKET2' : 0,
    'VOLCANIC_ROCK': 0,
    'VOLCANIC_ROCK_VOUCHER_10000': 0,
    'VOLCANIC_ROCK_VOUCHER_10250': 0,
    'VOLCANIC_ROCK_VOUCHER_10500': 0,
    'VOLCANIC_ROCK_VOUCHER_9500': 0,
    'VOLCANIC_ROCK_VOUCHER_9750': 0, 
}

cooldown_period = {
    'KELP' : 5, 
    'RAINFOREST_RESIN' : 5, 
    'SQUID_INK' : 5, 
    'JAMS' : 5, 
    'CROISSANTS' : 5, 
    'DJEMBES' : 5, 
    'PICNIC_BASKET1' : 5, 
    'PICNIC_BASKET2' : 5,
    'VOLCANIC_ROCK': 4,
    'VOLCANIC_ROCK_VOUCHER_10000': 5,
    'VOLCANIC_ROCK_VOUCHER_10250': 5,
    'VOLCANIC_ROCK_VOUCHER_10500': 5,
    'VOLCANIC_ROCK_VOUCHER_9500': 5,
    'VOLCANIC_ROCK_VOUCHER_9750': 5 
}

mid_price = {
    'KELP' : 0, 
    'RAINFOREST_RESIN' : 0, 
    'SQUID_INK' : 0, 
    'JAMS' : 0, 
    'CROISSANTS' : 0, 
    'DJEMBES' : 0, 
    'PICNIC_BASKET1' : 0, 
    'PICNIC_BASKET2' : 0,
    'VOLCANIC_ROCK': 0,
    'VOLCANIC_ROCK_VOUCHER_10000': 0,
    'VOLCANIC_ROCK_VOUCHER_10250': 0,
    'VOLCANIC_ROCK_VOUCHER_10500': 0,
    'VOLCANIC_ROCK_VOUCHER_9500': 0,
    'VOLCANIC_ROCK_VOUCHER_9750': 0  
}

######## BOLLINGER PARAMS ########
b_window_size = {
    'KELP' : 20, 
    'RAINFOREST_RESIN' : 20, 
    'SQUID_INK' : 10, 
    'JAMS' : 20, 
    'CROISSANTS' : 3, 
    'DJEMBES' : 20, 
    'PICNIC_BASKET1' : 3, 
    'PICNIC_BASKET2' : 3,
    'VOLCANIC_ROCK': 20,
    'VOLCANIC_ROCK_VOUCHER_10000': 20,
    'VOLCANIC_ROCK_VOUCHER_10250': 20,
    'VOLCANIC_ROCK_VOUCHER_10500': 20,
    'VOLCANIC_ROCK_VOUCHER_9500': 20,
    'VOLCANIC_ROCK_VOUCHER_9750': 20  
}  


######## EMA PARAMS ###########
ema_alpha = {
    'KELP' : 0.7, 
    'RAINFOREST_RESIN' : 0.7, ##
    'SQUID_INK' : 0.89, ##best value 0.89 for historycal
    'JAMS' : 0.8, ##best value 0.8 for hystorical
    'CROISSANTS' : 0.95, 
    'DJEMBES' : 0.5, 
    'PICNIC_BASKET1' : 0.5, 
    'PICNIC_BASKET2' : 0.5, 
    'VOLCANIC_ROCK': 0.9,
    'VOLCANIC_ROCK_VOUCHER_10000':0.9,
    'VOLCANIC_ROCK_VOUCHER_10250': 0.9,
    'VOLCANIC_ROCK_VOUCHER_10500': 0.9,
    'VOLCANIC_ROCK_VOUCHER_9500': 0.9,
    'VOLCANIC_ROCK_VOUCHER_9750': 0.9,
}  

###### ZSCORE PARAMS ##########
z_max_volume = {
    'KELP' : 25, 
    'RAINFOREST_RESIN' : 25, 
    'SQUID_INK' : 25, 
    'JAMS' : 25, 
    'CROISSANTS' : 25, 
    'DJEMBES' : 25, 
    'PICNIC_BASKET1' : 25, 
    'PICNIC_BASKET2' : 25, 
}

###### CROSSOVER PARAMS ######
cross_windows = {
    'KELP' : {'short' : 3, 'long' : 7}, 
    'RAINFOREST_RESIN' : {'short' : 3, 'long' : 7}, 
    'SQUID_INK' : {'short' : 3, 'long' : 7}, 
    'JAMS' : {'short' : 3, 'long' : 7}, 
    'CROISSANTS' : {'short' : 3, 'long' : 7}, 
    'DJEMBES' : {'short' : 3, 'long' : 7}, 
    'PICNIC_BASKET1' : {'short' : 3, 'long' : 7}, 
    'PICNIC_BASKET2' : {'short' : 3, 'long' : 7}, 
}

###### MOMENTUM PARAMS ######
mom_max_vol = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 10, 
    'SQUID_INK' : 10, 
    'JAMS' : 10, 
    'CROISSANTS' : 10, 
    'DJEMBES' : 10, 
    'PICNIC_BASKET1' : 10, 
    'PICNIC_BASKET2' : 10, 
}


###### FAIR_MM PARAMS ######
fmm_max_vol = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 10, 
    'SQUID_INK' : 10, 
    'JAMS' : 10, 
    'CROISSANTS' : 10, 
    'DJEMBES' : 10, 
    'PICNIC_BASKET1' : 10, 
    'PICNIC_BASKET2' : 10, 
}

###### TRENDFOLLOW PARAMS #######
tf_max_vol = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 10, 
    'SQUID_INK' : 10, 
    'JAMS' : 10, 
    'CROISSANTS' : 10, 
    'DJEMBES' : 10, 
    'PICNIC_BASKET1' : 10, 
    'PICNIC_BASKET2' : 10, 
}

###### ORDERBOOK IMBALANCE PARAMS #########
obimb_max_vol = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 10, 
    'SQUID_INK' : 10, 
    'JAMS' : 10, 
    'CROISSANTS' : 10, 
    'DJEMBES' : 10, 
    'PICNIC_BASKET1' : 10, 
    'PICNIC_BASKET2' : 10, 
}

imbalance_thresholds = {
    'KELP' : 0.3, 
    'RAINFOREST_RESIN' : 0.3, 
    'SQUID_INK' : 0.3, 
    'JAMS' : 0.3, 
    'CROISSANTS' : 0.3, 
    'DJEMBES' : 0.3, 
    'PICNIC_BASKET1' : 0.3, 
    'PICNIC_BASKET2' : 0.3, 
}

###### KELNER PARAMS #######
kel_max_vol = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 10, 
    'SQUID_INK' : 10, 
    'JAMS' : 10, 
    'CROISSANTS' : 10, 
    'DJEMBES' : 10, 
    'PICNIC_BASKET1' : 10, 
    'PICNIC_BASKET2' : 10, 
}

kel_atr_scale = {
    'KELP' : 1.5, 
    'RAINFOREST_RESIN' : 1.5, 
    'SQUID_INK' : 1.5, 
    'JAMS' : 1.5, 
    'CROISSANTS' : 1.5, 
    'DJEMBES' : 1.5, 
    'PICNIC_BASKET1' : 1.5, 
    'PICNIC_BASKET2' : 1.5, 
}


###### AVELLANEDA PARAMS ##########
trading_time = 1e6

gamma = {
    'KELP' : 1e-9, 
    'RAINFOREST_RESIN' : 1e-9, 
    'SQUID_INK' : 1e-9, 
    'JAMS' : 1e6, 
    'CROISSANTS' : 1e-9, 
    'DJEMBES' : 1e-9, 
    'PICNIC_BASKET1' : 1e-9, 
    'PICNIC_BASKET2' : 1e-9,
    'VOLCANIC_ROCK': 1e-9,
    'VOLCANIC_ROCK_VOUCHER_10000':1e-9,
    'VOLCANIC_ROCK_VOUCHER_10250': 1e-9,
    'VOLCANIC_ROCK_VOUCHER_10500': 1e-9,
    'VOLCANIC_ROCK_VOUCHER_9500': 1e-9,
    'VOLCANIC_ROCK_VOUCHER_9750': 1e-9, 
}

deviation_threshold = {
    'KELP' : 10, 
    'RAINFOREST_RESIN' : 45, 
    'SQUID_INK' : 10, 
    'JAMS' : 40, 
    'CROISSANTS' : 40, 
    'DJEMBES' : 30, 
    'PICNIC_BASKET1' : 40, 
    'PICNIC_BASKET2' : 40,
    'VOLCANIC_ROCK': 40,
    'VOLCANIC_ROCK_VOUCHER_10000': 40,
    'VOLCANIC_ROCK_VOUCHER_10250': 40,
    'VOLCANIC_ROCK_VOUCHER_10500': 40,
    'VOLCANIC_ROCK_VOUCHER_9500': 40,
    'VOLCANIC_ROCK_VOUCHER_9750': 40
     
}

prev_status = {
    'KELP' : 'normal', 
    'RAINFOREST_RESIN' : 'normal', 
    'SQUID_INK' : 'normal', 
    'JAMS' : 'normal', 
    'CROISSANTS' : 'normal', 
    'DJEMBES' : 'normal', 
    'PICNIC_BASKET1' : 'normal', 
    'PICNIC_BASKET2' : 'normal',
    'VOLCANIC_ROCK': 'normal',
    'VOLCANIC_ROCK_VOUCHER_10000': 'normal',
    'VOLCANIC_ROCK_VOUCHER_10250': 'normal',
    'VOLCANIC_ROCK_VOUCHER_10500': 'normal',
    'VOLCANIC_ROCK_VOUCHER_9500': 'normal',
    'VOLCANIC_ROCK_VOUCHER_9750': 'normal', 
}
ema={
    'KELP' : None, 
    'RAINFOREST_RESIN' : None, 
    'SQUID_INK' : None, 
    'JAMS' : None, 
    'CROISSANTS' : None, 
    'DJEMBES' : None, 
    'PICNIC_BASKET1' : None, 
    'PICNIC_BASKET2' : None,
    'VOLCANIC_ROCK': None,
    'VOLCANIC_ROCK_VOUCHER_10000': None,
    'VOLCANIC_ROCK_VOUCHER_10250': None,
    'VOLCANIC_ROCK_VOUCHER_10500': None,
    'VOLCANIC_ROCK_VOUCHER_9500': None,
    'VOLCANIC_ROCK_VOUCHER_9750': None,  
}
############# PAIR TRADING PARAMS #########
spread_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=20)))


############# END OF PARAMS ###############

products = [ 'RAINFOREST_RESIN', 'KELP','PICNIC_BASKET1','PICNIC_BASKET2']

# products = [ 'SQUID_INK']
# products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK' ]
# products = ['PICNIC_BASKET1','PICNIC_BASKET2' ]
# products = []
# pairs = [( 'JAMS' , 'CROISSANTS')]
# pairs = [('PICNIC_BASKET1', 'PICNIC_BASKET2')]    
# products.extend(['VOLCANIC_ROCK_VOUCHER_10000','VOLCANIC_ROCK_VOUCHER_9750','VOLCANIC_ROCK_VOUCHER_9500','VOLCANIC_ROCK_VOUCHER_10250','VOLCANIC_ROCK_VOUCHER_10500'])
trips = [('JAMS', 'DJEMBES', "CROISSANTS")]
# vouchers =[]
# trips = []
pairs = []
class Trader:
    def __init__(self):
        self.params=  {'SQUID_INK':{
        "mean_reversion_beta": -0.45,
        "take_width": 1.5,
        "clear_width": 0.75,
        "adverse_volume": 20,
        "min_edge": 4,
        "spread_window": 50,
    }
    
}
        self.TTE_MAX = 4
        """ Storing all the params locally for speed """
        self.renko_history = {}  # per-product brick direction history
        self.last_renko_price = {}  # per-product base renko level
        self.strike_price = {
                'VOLCANIC_ROCK_VOUCHER_10000': 10000,
                'VOLCANIC_ROCK_VOUCHER_10250': 10250,
                'VOLCANIC_ROCK_VOUCHER_10500': 10500,
                'VOLCANIC_ROCK_VOUCHER_9500': 9500,
                'VOLCANIC_ROCK_VOUCHER_9750': 9750,  
        }
        self.kelpres_maxed = defaultdict(lambda :deque(maxlen=20))
        self.vol_hist = defaultdict(lambda: deque(maxlen=20))
        self.products = list(products)
        self.product_params = {'max_position' : dict(max_position), 
                               'ema':dict(ema),
                               'valuation_strategy':dict(valuation_strategy),
                               'b_window_size' : dict(b_window_size), 
                               'ema_alpha' : dict(ema_alpha), 
                               'z_max_volume' : dict(z_max_volume), 
                               'cross_windows' : dict(cross_windows), 
                               'mom_max_vol' : dict(mom_max_vol), 
                               'fmm_max_vol' : dict(fmm_max_vol), 
                               'tf_max_vol' : dict(tf_max_vol), 
                               'obimb_max_vol' : dict(obimb_max_vol), 
                               'imbalance_thresholds' : dict(imbalance_thresholds), 
                               'kel_max_vol' : dict(kel_max_vol), 
                               'kel_atr_scale' : dict(kel_atr_scale), 
                               'trading_time' : trading_time, 
                               'price_history' : price_history, 
                               'cooldown' : dict(cooldown), 
                               'cooldown_period' : dict(cooldown_period), 
                               'mid_price' : dict(mid_price), 
                               'gamma' : dict(gamma), 
                               'alpha' : dict(ema_alpha), 
                               'deviation_threshold' : dict(deviation_threshold), 
                               'prev_status' : dict(prev_status),
                               'spread_history' : spread_history, 

                               }
        self.product_strategy = {
                'KELP' : 'KELPRESIN',              ##curentbest marketmakerrape
                'RAINFOREST_RESIN' : 'KELPRESIN',  ##curentbest marketmakerrape
                'SQUID_INK' : 'KELPRESIN',         ##curentbest BOLLINGER
                'JAMS' : 'RENKO',              ##curentbest IDK
                'CROISSANTS' : 'RENKO',        ##curentbest IDK
                'DJEMBES' : 'RENKO',           ##curentbest IDK
                'PICNIC_BASKET1' : 'KELPRESIN',    ##curentbest IDK
                'PICNIC_BASKET2' : 'KELPRESIN',
                'VOLCANIC_ROCK': 'AVELLANEDA',
                'VOLCANIC_ROCK_VOUCHER_10000': 'AVELLANEDA',
                'VOLCANIC_ROCK_VOUCHER_10250': 'AVELLANEDA',
                'VOLCANIC_ROCK_VOUCHER_10500': 'AVELLANEDA',
                'VOLCANIC_ROCK_VOUCHER_9500': 'AVELLANEDA',
                'VOLCANIC_ROCK_VOUCHER_9750': 'AVELLANEDA',    ##curentbest IDK
            }
        self.strategy = {
            'AVELLANEDA' : self.avellaneda,
            'BOLLINGER' : self.bollinger_strategy,
            # 'BREAKOUT' : self.breakout_strategy,
            # 'MOVEAVG' : self.moving_average_strategy,
            # 'ZSCORE' : self.zscore_strategy,
            # 'CROSSOVER' : self.crossover_strategy,
            # 'MOMENTUM' : self.momentum_strategy,
            # 'FAIRPRICE' : self.fair_price_mm_strategy,
            # 'IMBALANCE' : self.orderbook_imbalance_strategy,
            # 'KELTNER' : self.keltner_channel_strategy, 
            # 'MMCOPY' : self.market_maker_strategy,
            'KELPRESIN' : self.kelp_strat, 
            'RENKO' : self.renko_strategy
        }
    
    def run(self, state: TradingState):
        result = {}
        # conversions = []
        for product in products:
            if product not in state.listings:
                continue
            if product == 'PICNIC_BASKET1' or product == 'PICNIC_BASKET2':
                order_dict = self.basket_arb(product, state)
                for com , order in order_dict.items():
                    if len(order) > 0:
                        result[com] = order
            else:
                order = self.strategy[self.product_strategy[product]](product, state.order_depths[product], state.position.get(product,0), state.timestamp)
                if len(order) > 0:
                    result[product] = order

        for prod1, prod2 in pairs:
            if prod1 not in state.listings:
                continue
            if prod2 not in state.listings:
                continue
            order_dict = self.pair_trading(prod1,prod2, state)
            for prod, order in order_dict.items():
                result[prod] = order
            # conversions.append(converts)
        for prod1, prod2, prod3 in trips:
            # logger.print("sex")
            order_dict=self.trip_strategy(prod1,prod2,prod3,state)
            for prod, order in order_dict.items():
                if len(order) > 0:
                   result[prod] = order
        order_dict = self.smail_arb(state)
        for prod, order in order_dict.items():
                    if len(order) > 0:
                        result[prod] = order
        logger.flush(state, result, 0, "")
        return result, 0, ""



    ######### STRATEGIES #######

    def avellaneda(self, product, order_depth, current_position, timestamp):
        orders = []
        

        mid_price, best_ask ,best_bid=self.get_mid_price(product,order_depth)

        self.product_params['price_history'][product].append(mid_price)
        realized_vol = self.calculate_volatility(self.product_params['price_history'][product])
        effective_sigma = realized_vol
        
        k = self.calculate_k(order_depth, mid_price)
        gamma = max(self.product_params['gamma'][product]/(1+ 20 * realized_vol), 1e-10)

        # Spread calculation with dynamic volatility
        # time_left = (self.product_params['trading_time'] - timestamp)/self.product_params['trading_time']
        spread = (2/gamma) * math.log(1 + gamma/k)            
        market_spread = best_ask - best_bid

        rest_price = mid_price - current_position * gamma * (effective_sigma**2) 
        
        bid_price = int(rest_price - spread/2)
        ask_price = int(rest_price + spread/2)

        # inventory_ratio = abs(current_position) / self.product_params['max_position'][product]
        # aggression_factor = 1 + 3.5*inventory_ratio  # ranges from 1 to 3

        # 2. Add new limit orders if no matches
        num_levels = 3 # Number of price levels
        level_spacing = min(spread , market_spread)/(2*num_levels)
        
        # Clear existing positions if needed
        # print(self.product_params['max_position'][product])
        remaining_buy = self.product_params['max_position'][product] - current_position
        remaining_sell = self.product_params['max_position'][product] + current_position

        status, ewma_mid = self.detect_fake(product, mid_price)

        if status == "cooldown":
            # logger.print(f"{product} in cooldown — not quoting")
            return [] # Don't quote during cooldown

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

                return orders


        # 1. Take existing liquidity
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask <= bid_price:
                max_buy = min(self.product_params['max_position'][product] - current_position, -vol)
                if max_buy > 0:
                    orders.append(Order(product, ask, max_buy))
                    remaining_buy -= max_buy
                    current_position += max_buy

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= ask_price:
                max_sell = min(self.product_params['max_position'][product] + current_position, vol)
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
                level_factor = ((num_levels - i) / num_levels)**0.1
                bid_size = int(remaining_buy * level_factor / num_levels)
                ask_size = int(remaining_sell * level_factor / num_levels)
                
                # Place orders if size > 0
                if bid_size > 0:
                    orders.append(Order(product, bid_level_price, bid_size))
                if ask_size > 0:
                    orders.append(Order(product, ask_level_price, -ask_size))

        # logger.print("-----------------------",bid_price, ask_price, rest_price,k,"------------------------")
    
        return orders
        
    def bollinger_strategy(self, product, order_depth, current_position, timestamp):
        p = self.product_params
        orders = []
        mid_price, best_ask, best_bid=self.get_mid_price(product,order_depth)


        if len(p['price_history'][product]) < p['b_window_size'][product]:
            return []

        prices = list(p['price_history'][product])
        std = np.std(prices)
        spread = std * (1.4 + std / 5)
        market_spread = best_ask - best_bid
        level_spacing = min(spread, market_spread) / 6  # 3 levels x 2 sides

        remaining_buy = p['max_position'][product] - current_position
        remaining_sell = p['max_position'][product] + current_position

        status, ewma_mid = self.detect_fake(product, mid_price)
        # logger.print(mid_price, ewma_mid, status)

        # --- Trend check ---
        # trend = mid_price - ewma_mid
        # trend_threshold = 2 * std  # adjustable

        # disable_buy = trend < -trend_threshold and current_position >= 0
        # disable_sell = trend > trend_threshold and current_position <= 0

        if status == "cooldown":
            # logger.print(f"{product} in cooldown — not quoting")
            return []

        # elif status == "fake":
            # logger.print(f"{product} FAKE detected at {mid_price}, smoothed {ewma_mid}")

            # # Only fade fake if price is NOT in strong trend
            # if trend < trend_threshold and mid_price < ewma_mid and remaining_buy > 0:
            #     fade_price = int(mid_price)
            #     size = int(remaining_buy * 0.4)  # partial fade
            #     orders.append(Order(product, fade_price, size))

            # elif trend > -trend_threshold and mid_price > ewma_mid and remaining_sell > 0:
            #     fade_price = int(mid_price)
            #     size = int(remaining_sell * 0.4)
            #     orders.append(Order(product, fade_price, -size))

            # return orders

        elif status == "snap_back":
            # logger.print(f"{product} SNAPBACK at {mid_price}, smoothed {ewma_mid}")

            # Direction-aware snap trading
            if mid_price < ewma_mid and remaining_sell > 0:
                # Rebound expected UP → sell into it
                rebound_price = int(ewma_mid + 1)
                size = int(remaining_sell * 0.5)
                orders.append(Order(product, rebound_price, -size))

            elif mid_price > ewma_mid and remaining_buy > 0:
                # Rebound expected DOWN → buy into it
                rebound_price = int(ewma_mid - 1)
                size = int(remaining_buy * 0.5)
                orders.append(Order(product, rebound_price, size))

            return orders

        # # --- Default quoting path (normal mode) ---
        rest_price = 0.4 * mid_price + 0.6 * ewma_mid
        bid_price = int(rest_price - spread / 2)
        ask_price = int(rest_price + spread / 2)

        # # Avoid quoting aggressively in strong trend
        # if trend > trend_threshold and current_position <= 0:
        #     # logger.print("Uptrend /detected — not quoting sell side")
        # ask_price = float('inf')  # disable sell

        # if trend < -trend_threshold and current_position >= 0:
        #     # logger.print("Downtrend detected — not quoting buy side")
        # bid_price = 0  # disable buy

        # --- Take existing liquidity ---
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask <= bid_price:
                max_buy = min(remaining_buy, -vol)
                if max_buy > 0:
                    orders.append(Order(product, ask, max_buy))
                    remaining_buy -= max_buy
                    current_position += max_buy

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= ask_price:
                max_sell = min(remaining_sell, vol)
                if max_sell > 0:
                    orders.append(Order(product, bid, -max_sell))
                    remaining_sell -= max_sell
                    current_position -= max_sell

        # --- Place layered passive quotes ---
        num_levels = 3
        for i in range(num_levels):
            bid_level_price = int(rest_price - (i + 1) * level_spacing)
            ask_level_price = int(rest_price + (i + 1) * level_spacing)

            level_factor = ((num_levels - i) / num_levels) ** 2  # more weight inside
            bid_size = int(remaining_buy * level_factor / num_levels)
            ask_size = int(remaining_sell * level_factor / num_levels)

            # if ask_size > 0 and not disable_sell:
            orders.append(Order(product, ask_level_price, -ask_size))
            # if bid_size > 0 and not disable_buy:
            orders.append(Order(product, bid_level_price, bid_size))

        self.product_params['prev_status'][product] = status
        return orders

    def kelp_strat(self, product ,order_depth, current_position, timestamp):
        orders = []

        mid_price, best_ask, best_bid = self.get_mid_price(product, order_depth)
        if abs(current_position) == self.product_params['max_position'][product]:
            self.kelpres_maxed[product].append(1)
        else:
            self.kelpres_maxed[product].append(0)

        remaining_buy = min(self.product_params['max_position'][product] ,self.product_params['max_position'][product] - current_position)
        remaining_sell = min(self.product_params['max_position'][product] ,self.product_params['max_position'][product] + current_position)

        min_ask = min(order_depth.sell_orders.keys())
        max_buy = max(order_depth.buy_orders.keys())

        # if abs(current_position) == self.product_params['max_position'][product]:
        #     orders.append(Order(product, mid_price, -current_position))
        #     return orders
        # if max_buy+1 <=mid_price:
        # if sum(list(self.kelpres_maxed[product])) >=20:
        #     orders.append(Order(product, mid_price, -current_position))
        #     return orders

        orders.append(Order(product, min(max_buy+1,mid_price), remaining_buy))
        # elif current_position == -self.product_params['max_position'][product]:
        #     orders.append(Order(product, mid_price, -current_position))
        # if min_ask-1 >=mid_price:
        orders.append(Order(product, max(min_ask-1, mid_price), -remaining_sell))
        # elif current_position == self.product_params['max_position'][product]:
            # orders.append(Order(product, mid_price, -current_position))


        return orders


    def renko_strategy(self, product, order_depth, current_position, timestamp):
        orders = []
        p = self.product_params
        
        # Setup state
        brick_size = p.get('brick_size', 10)
        confirmation_bricks = p.get('confirmation_bricks', 4)
        max_position = p['max_position'][product]

        mid_price, best_ask, best_bid = self.get_mid_price(product, order_depth)

        # Initialize tracking
        if product not in self.renko_history:
            self.renko_history[product] = deque(maxlen=20)
            self.last_renko_price[product] = mid_price

        # if len(self.renko_history[product]) < 10:
        #     return []


        bricks = self.renko_history[product]
        last_renko_price = self.last_renko_price[product]
        price_diff = mid_price - last_renko_price

        direction = 0
        if abs(price_diff) >= brick_size:
            direction = int(math.copysign(1, price_diff))
            num_bricks = int(abs(price_diff) // brick_size)
            for _ in range(num_bricks):
                bricks.append(direction)
                last_renko_price += direction * brick_size
            self.last_renko_price[product] = last_renko_price

        # Not enough bricks to act
        if len(bricks) < confirmation_bricks:
            return []

        # Check if we have strong directional trend
        recent_trend = list(bricks)[-confirmation_bricks:]
        trend_direction = bricks[-1]
        if all(b == trend_direction for b in recent_trend):
            # Trend confirmed
            remaining_buy = max_position - current_position
            remaining_sell = max_position + current_position

            if trend_direction == 1 and remaining_buy > 0:
                # Uptrend → Buy aggressively
                buy_price = best_ask 
                orders.append(Order(product, buy_price, remaining_buy))

            elif trend_direction == -1 and remaining_sell > 0:
                # Downtrend → Sell aggressively
                sell_price = best_bid 
                orders.append(Order(product, sell_price, -remaining_sell))

        return orders



    def basket_arb(self, basket, state : TradingState):
        order_dict = defaultdict(list)
        # baskets = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
        baskets = {'PICNIC_BASKET1' : {'CROISSANTS' : 6, 'DJEMBES' : 1 , 'JAMS' : 3}, 'PICNIC_BASKET2' : {'CROISSANTS' : 4, 'JAMS' : 2}}
        min_asks = {}
        max_bids = {}
        mids = {}
        
        materials = baskets[basket]
        synthetic_price = 0
        for mat, wei in materials.items():
            mid_price, best_ask, best_bid = self.get_mid_price(mat, state.order_depths[mat])
            synthetic_price += mid_price*wei

            mids[mat] = mid_price
            min_asks[mat] = min(state.order_depths[mat].sell_orders.keys())
            max_bids[mat] = max(state.order_depths[mat].buy_orders.keys())
        
        real_price , pb_ask, pb_bid = self.get_mid_price(basket, state.order_depths[basket])

        # mat_max_buy = min(self.product_params['max_position'][mat], int(min((self.product_params['max_position'][mat] - state.position.get(mat, 0))/wei for mat,wei in materials.items())))
        # mat_max_sell = min(self.product_params['max_position'][mat], int(min((self.product_params['max_position'][mat] + state.position.get(mat, 0))/wei for mat,wei in materials.items())))

        pb_max_buy = min(self.product_params['max_position'][basket], self.product_params['max_position'][basket] - state.position.get(basket, 0))
        pb_max_sell = min(self.product_params['max_position'][basket], self.product_params['max_position'][basket] + state.position.get(basket, 0))

        pb_min_ask = min(state.order_depths[basket].sell_orders.keys())
        pb_max_bid = max(state.order_depths[basket].buy_orders.keys())

        threshold = np.std(list(self.product_params['price_history'][basket]))

    

        if abs(synthetic_price - real_price) >=threshold:
            if synthetic_price  > real_price:
                order_dict[basket].append(Order(basket, min(pb_max_bid+1, real_price-1), pb_max_buy))

                # for mat, wei in materials.items():
                #     order_dict[mat].append(Order(mat, mids[mat], -mat_max_sell*wei))
    
            else:
                order_dict[basket].append(Order(basket, max(pb_min_ask-1, real_price+1), -pb_max_sell))

                # for mat, wei in materials.items():
                #     order_dict[mat].append(Order(mat, mids[mat], mat_max_buy*wei))
        return order_dict

    def pair_trading(self, product1, product2, state : TradingState):

        order_dict = defaultdict(list)

        p1_mid, p1_best_ask, p1_best_bid = self.get_mid_price(product1, state.order_depths[product1])
        p2_mid, p2_best_ask, p2_best_bid = self.get_mid_price(product2, state.order_depths[product2])
        
        p1_max_buy = min(self.product_params['max_position'][product1], self.product_params['max_position'][product1] - state.position.get(product1, 0))
        p1_max_sell = min(self.product_params['max_position'][product1], self.product_params['max_position'][product1] + state.position.get(product1, 0))

        p2_max_buy = min(self.product_params['max_position'][product2], self.product_params['max_position'][product2] - state.position.get(product2, 0))
        p2_max_sell = min(self.product_params['max_position'][product2], self.product_params['max_position'][product2] + state.position.get(product2, 0))

        if min(len(self.product_params['price_history'][product1]),len(self.product_params['price_history'][product1])) <10:
            return order_dict
        beta = self.OLS_estimate_beta(product1, product2)
        
        # p1_mid -=

        spread = p1_mid - beta*p2_mid
        self.product_params['spread_history'][product1][product2].append(spread)
        if len(self.product_params['spread_history'][product1][product2]) < 10:
            return order_dict
        spread_threshold = np.std(list(self.product_params['spread_history'][product1][product2]))*(1+ 20*self.calculate_volatility(self.product_params['spread_history'][product1][product2]))

        if abs(spread) >= spread_threshold:
            if spread > 0:
                order_dict[product1].append(Order(product1, p1_mid , -p1_max_sell))
            
                order_dict[product2].append(Order(product2, p2_mid , p2_max_buy))
            else:
                order_dict[product1].append(Order(product1, p1_mid , p1_max_buy))
            
                order_dict[product2].append(Order(product2, p2_mid , -p2_max_sell))
            
        return order_dict

    def trip_strategy(self, product1, product2,product3, state : TradingState):
        # with open(rf'C:\Users\shand\OneDrive\Desktop\IMC\imc-prosperity-3-backtester\betas.txt','a') as f:
        #     f.close()
        order_dict = defaultdict(list)
        # p1_mid, p1_best_ask, p1_best_bid = self.get_mid_price(product1, state.order_depths[product1])
        # p2_mid, p2_best_ask, p2_best_bid = self.get_mid_price(product2, state.order_depths[product2])
        # p3_mid, p3_best_ask, p3_best_bid = self.get_mid_price(product3, state.order_depths[product3])
        
        p1_mid = self.product_params['price_history'][product1][-1]
        p2_mid = self.product_params['price_history'][product2][-1]
        p3_mid = self.product_params['price_history'][product3][-1]

        p1_best_ask = min(state.order_depths[product1].sell_orders.keys())
        p1_best_bid = max(state.order_depths[product1].buy_orders.keys())

        p2_best_ask = min(state.order_depths[product2].sell_orders.keys())
        p2_best_bid = max(state.order_depths[product2].buy_orders.keys())

        p3_best_ask = min(state.order_depths[product3].sell_orders.keys())
        p3_best_bid = max(state.order_depths[product3].buy_orders.keys())

        p1_mid=np.mean(list(self.product_params['price_history'][product1])[-10:])
        p2_mid=np.mean(list(self.product_params['price_history'][product2])[-10:])
        p3_mid=np.mean(list(self.product_params['price_history'][product3])[-10:])
        
        p1_max_buy = min(self.product_params['max_position'][product1], self.product_params['max_position'][product1] - state.position.get(product1, 0))
        p1_max_sell = min(self.product_params['max_position'][product1], self.product_params['max_position'][product1] + state.position.get(product1, 0))

        p2_max_buy = min(self.product_params['max_position'][product2], self.product_params['max_position'][product2] - state.position.get(product2, 0))
        p2_max_sell = min(self.product_params['max_position'][product2], self.product_params['max_position'][product2] + state.position.get(product2, 0))
        
        p3_max_buy = min(self.product_params['max_position'][product3], self.product_params['max_position'][product3] - state.position.get(product3, 0))
        p3_max_sell = min(self.product_params['max_position'][product3], self.product_params['max_position'][product3] + state.position.get(product3, 0))

        # logger.print(p1_mid,p2_mid,p3_mid)
        # logger.print(min(len(self.product_params['price_history'][product1]),len(self.product_params['price_history'][product2]),len(self.product_params['price_history'][product3])))
        if min(len(self.product_params['price_history'][product1]),len(self.product_params['price_history'][product2]),len(self.product_params['price_history'][product3])) <10:
            return order_dict
        
        beta12 = self.OLS_estimate_beta(product1, product2)
        beta23 = self.OLS_estimate_beta(product2, product3)
        beta13 = self.OLS_estimate_beta(product1, product3)
                
        # p1_mid -=

        spread12 = p1_mid*(1-0.0001*(state.position.get(product1, 0)/self.product_params['max_position'][product1])) - beta12*p2_mid*((1-0.0001*(state.position.get(product2, 0)/self.product_params['max_position'][product2])))
        spread23 = p2_mid*(1-0.0001*(state.position.get(product2, 0)/self.product_params['max_position'][product2])) - beta23*p3_mid*(1-0.01*(state.position.get(product3, 0)/self.product_params['max_position'][product3]))
        spread13 = p1_mid*((1-0.0001*(state.position.get(product1, 0)/self.product_params['max_position'][product1]))) - beta13*p3_mid*(1-0.01*(state.position.get(product3, 0)/self.product_params['max_position'][product3]))
        

        # with open(rf'C:\Users\shand\OneDrive\Desktop\IMC\imc-prosperity-3-backtester\betas.txt','a') as f:
        #     f.write(f"{state.timestamp}\n")
        #     f.write(f'spread12: {spread12}, beta 12:{beta12}\n')
        #     f.write(f'correlation of 1,2: {self.calculate_correlation(product1,product2,20)}\n')
        #     f.write(f'spread23: {spread23}, beta 23:{beta23}\n')
        #     f.write(f'spread13: {spread13}, beta 13:{beta13}\n\n')
        #     f.close()
        logger.print("meao")
        logger.print(spread12,beta12)
        logger.print(spread23,beta23)
        logger.print(spread13,beta13)
        
        
        
        self.product_params['spread_history'][product1][product2].append(spread12)
        self.product_params['spread_history'][product2][product3].append(spread23)
        self.product_params['spread_history'][product3][product1].append(spread13)  
        # logger.print(self.product_params['spread_history'][product1][product2])
        # logger.print(self.product_params['spread_history'][product2][product3])
        # logger.print(self.product_params['spread_history'][product3][product1])

        # if len(self.product_params['spread_history'][product1][product2]) or len(self.product_params['spread_history'][product2][product3]) or len(self.product_params['spread_history'][product3][product1]) < 10:
        #     return order_dict
        
        spread_threshold12 = 2.5*np.std(list(self.product_params['spread_history'][product1][product2]))
        spread_threshold23 = 3.5*np.std(list(self.product_params['spread_history'][product2][product3]))
        spread_threshold13 = 3.5*np.std(list(self.product_params['spread_history'][product3][product1]))

        # co12=self.calculate_correlation(product1,product2,20)
        # co23=self.calculate_correlation(product3,product2,20)
        # co13=self.calculate_correlation(product1,product3,20)
        
        # spread_threshold12 = spread_threshold23 = spread_threshold13 = 15
        
        # logger.print(np.corrcoef(product1,product2))
        # 3. Johansen cointegration test
        # min_history = 60
        # prices = np.vstack([
        #     list(self.product_params['price_history'][product1])[-min_history:],
        #     list(self.product_params['price_history'][product2])[-min_history:],
        #     list(self.product_params['price_history'][product3])[-min_history:]
        # ]).T
        
        # if not self.johansen_cointegration_test(prices):
        #     return order_dict

        if spread12>=spread_threshold12 and spread23<= -spread_threshold23:            
                order_dict[product2].append(Order(product2, p2_best_ask , p2_max_buy>>1))
        elif spread12 <=-spread_threshold12 and spread23>= spread_threshold23:            
                order_dict[product2].append(Order(product2, p2_best_bid , -p2_max_sell>>1))
        if spread23>=spread_threshold23 and spread13<= -spread_threshold13:            
                order_dict[product3].append(Order(product3, p3_best_ask , p3_max_buy>>1))
        elif spread23<=-spread_threshold23 and spread13>= spread_threshold13:            
                order_dict[product3].append(Order(product3, p3_best_bid , -p3_max_sell>>1))
        if spread13>=spread_threshold13 and spread12>= spread_threshold12:            
                order_dict[product1].append(Order(product1, p1_best_ask , p1_max_buy>>1))
        elif spread13<=-spread_threshold13 and spread12<= -spread_threshold12:            
                order_dict[product1].append(Order(product1, p1_best_bid , -p1_max_sell>>1))

        # logger.print(spread12,beta12)
        # logger.print(spread23,beta23)
        # logger.print(spread13,beta13)
        return order_dict

    def smail_arb(self, state: TradingState):
        # Smile curve coefficients
        a_t = 1.59195 
        b_t = 0.00258 
        c_t = 0.02210

        order_dict = defaultdict(list)
        net_delta = 0

        # Time to expiry in years normalized by TTE_MAX
        tte_raw = (self.TTE_MAX - state.timestamp / 1e6) / 252
        tte = max(tte_raw / self.TTE_MAX, 0.05)  # Avoid tte being too small

        # VOLCANIC_ROCK price and position
        rock_mid, rock_best_ask, rock_best_bid = self.get_mid_price('VOLCANIC_ROCK', state.order_depths['VOLCANIC_ROCK'])
        if rock_mid is None or rock_mid <= 0:
            logger.print("Skipping due to invalid VOLCANIC_ROCK mid price")
            return order_dict

        rock_pos = state.position.get('VOLCANIC_ROCK', 0)
        max_pos = self.product_params['max_position']['VOLCANIC_ROCK']
        rock_max_buy = min(max_pos,max_pos - rock_pos)
        rock_max_sell = min(max_pos,max_pos + rock_pos)

        for voucher, strike in self.strike_price.items():
            # vou_mid, vou_best_ask, vou_best_bid = self.get_mid_price(voucher, state.order_depths[voucher])
            if len(self.product_params['price_history'][voucher]) == 0:
                continue
            vou_mid  = list(self.product_params['price_history'][voucher])[-1]

            if vou_mid is None:
                continue

            current_position = state.position.get(voucher, 0)
            max_pos = self.product_params['max_position'][voucher]
            vou_max_buy = max_pos - current_position
            vou_max_sell = max_pos + current_position

            # Calculate true IV
            true_vol = self.implied_volatility(vou_mid, rock_mid, strike, tte)

            # Smile-based fitted IV
            try:
                m_t = log(strike / rock_mid) / sqrt(tte)
            except ValueError:
                continue  # Skip if log() or sqrt() fails

            synth_vol = a_t * m_t**2 + b_t * m_t + c_t
            self.vol_hist[voucher].append(true_vol)

            if len(self.vol_hist[voucher]) < 10:
                continue

            smu_vol = np.mean(self.vol_hist[voucher])
            vol_thresh = np.std(self.vol_hist[voucher])
            logger.print(f"{voucher}: true_vol={true_vol:.4f}, fitted_vol={synth_vol:.4f}, smu_vol={smu_vol:.4f}, thresh={vol_thresh:.4f}")

            # Smile-based delta
            smile_delta = self.bs_call_delta(rock_mid, strike, tte, synth_vol)

            # Trade if mispricing detected
            if abs(smu_vol - synth_vol) > vol_thresh:
                if smu_vol > synth_vol and vou_max_sell > 0:
                    qty = (vou_max_sell)>>1  # limit trade size for safety
                    net_delta -= qty * smile_delta
                    # order_dict[voucher].append(Order(voucher, vou_best_bid, -qty))
                elif smu_vol < synth_vol and vou_max_buy > 0:
                    qty = (vou_max_buy)>>1
                    net_delta += qty * smile_delta
                    # order_dict[voucher].append(Order(voucher, vou_best_ask, qty))

        # Hedge with VOLCANIC_ROCK
        hedge_qty = int(round(-net_delta))
        logger.print(f"Net delta from options: {net_delta:.2f}, hedging with {hedge_qty} VOLCANIC_ROCK")

        if hedge_qty > 0  and rock_best_ask is not None:
            order_dict['VOLCANIC_ROCK'].append(Order('VOLCANIC_ROCK', rock_best_ask, min(hedge_qty, rock_max_buy)))
        elif hedge_qty < 0 and rock_best_bid is not None:
            order_dict['VOLCANIC_ROCK'].append(Order('VOLCANIC_ROCK', rock_best_bid, max(hedge_qty, -rock_max_sell)))

        return order_dict


# def 

    # Trader.avellaneda = avellaneda
    ####### END OF STRATEGIES #########













    ###### ALL UTILITY FUNCTIONS #######

    def norm_cdf(self,x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    
    def bs_call_delta(self,S, K, T, sigma):
        if T <= 0 or sigma <= 0:
            return 0.5  # flat guess
        d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
        return self.norm_cdf(d1)
    
    def norm_cdf(self,x: float) -> float:
        # Standard normal CDF using error function
        return 0.5 * (1 + erf(x / sqrt(2)))

    def black_scholes_price(self,S: float, K: float, T: float, sigma: float)-> float:
        if T == 0 or sigma == 0:
            return max(0.0, S - K) 

        d1 = (log(S / K) + (0.06 + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        return S * self.norm_cdf(d1) - K * exp(-0.06 * T) * self.norm_cdf(d2)
        
    def implied_volatility(self,
        market_price: float,
        S: float, #volkPrice
        K: float, #strike
        T: float, #TTE
        tol: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        # Use binary search between 0.0001 and 5.0 vol
        low = 0.0001
        high = 5.0
        for i in range(max_iterations):
            mid = (low + high) / 2
            price = self.black_scholes_price(S, K, T, mid)

            if abs(price - market_price) < tol:
                return mid
            elif price > market_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2  # best guess

    
    def get_mid_price(self, product, order_depth, window_size=3):
        strategy = self.product_params['valuation_strategy'][product]
        #print(strategy)
        # if strategy == 'true_value' or strategy == 'mid':
        if order_depth.buy_orders:
            max_bid_volume = max(order_depth.buy_orders.values())
            candidate_bids = {price: vol for price, vol in order_depth.buy_orders.items() if vol == max_bid_volume}
            best_bid = max(candidate_bids)  # Highest price among max volume bids
        else:
            best_bid = 0
            candidate_bids = {}

        if order_depth.sell_orders:
            max_ask_volume = max(abs(vol) for vol in order_depth.sell_orders.values())
            candidate_asks = {price: vol for price, vol in order_depth.sell_orders.items() if abs(vol) == max_ask_volume}
            best_ask = min(candidate_asks)  # Lowest price among max volume asks
        else:
            best_ask = float('inf')
            candidate_asks = {}

        if best_ask == float('inf') or best_bid == 0:
            mid_price
        mid_price = (best_bid + best_ask) >> 1

        if strategy == 'vwap':
            bid_prices = sorted(candidate_bids.keys(), reverse=True)
            bid_volumes = [candidate_bids[p] for p in bid_prices]
            ask_prices = sorted(candidate_asks.keys())
            ask_volumes = [abs(candidate_asks[p]) for p in ask_prices]

            vwap_bid = self.calculate_vwap(bid_prices, bid_volumes, best_bid)
            vwap_ask = self.calculate_vwap(ask_prices, ask_volumes, best_ask)
            mid_price = (vwap_bid + vwap_ask) >>1

        elif strategy == 'ema':
            alpha = 2 / (window_size + 1)
            if self.product_params['ema'][product] is None:
                self.product_params['ema'][product] = mid_price
            else:
                self.product_params['ema'][product] = alpha * mid_price + (1 - alpha) * self.product_params['ema'][product]
            mid_price = self.product_params['ema'][product]


        # if strategy == 'mid':

        self.product_params['price_history'][product].append(mid_price)
        return mid_price, best_ask , best_bid 
    
    def OLS_estimate_beta(self, prod1, prod2):
        l = min(len(self.product_params['price_history'][prod1]), len(self.product_params['price_history'][prod2]))
        A = np.array(list(self.product_params['price_history'][prod1])[:l])
        B = np.array(list(self.product_params['price_history'][prod2])[:l])
        beta = np.dot(A, B) / np.dot(B, B)
        return beta

    def vol_adj_beta(self, prod1, prod2):
        vol1 = self.calculate_volatility(self.product_params['price_history'][prod1])
        vol2 = self.calculate_volatility(self.product_params['price_history'][prod2])

        return vol1/vol2
    
    def calculate_vwap(self, prices, volumes, fallback_price):
        total_volume = sum(volumes)
        if total_volume == 0:
            return fallback_price  # fallback in case there's no volume
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return vwap


    def calculate_volatility(self, price_history: deque) -> float:              
        """Calculates volatility / sigma parameter on the fly"""
        if len(price_history) < 2:
            return 0  # Not enough data
        
        returns = []
        prev_price = price_history[0]
        for price in list(price_history)[1:]:
            if price & prev_price:
                returns.append(math.log(price / prev_price))
                prev_price = price
            
        if len(returns) < 1:
            return 0
            
        std_dev = np.std(returns)
        return std_dev

    def calculate_k(self, order_depth: OrderDepth, mid_price: float) -> float:
        """ Calculates OrderBook Depth / K Param on the fly"""
        
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

    def detect_fake(self, product, mid_price): 
        """ Detects a fake crash / drop like in round 1 day 1 dataset 

            Also returns the smoothed out mid_price for safe algo logic     
        
            """
        smoothed = self.product_params['mid_price'][product]
        ewma_mid = self.product_params['alpha'][product] * mid_price + (1 - self.product_params['alpha'][product]) * smoothed
        self.product_params['mid_price'][product] = ewma_mid
        
        deviation = abs(mid_price - ewma_mid)
        # logger.print("------", product, deviation, self.product_params['deviation_threshold'][product],deviation > self.product_params['deviation_threshold'][product],"-------")
        # Cooldown active? Don't quote
        # if self.product_params['prev_status'][product] == "fake" and abs(mid_price - ewma_mid) < std:
        #     return "snap_reversion"

        if self.product_params['cooldown'][product] > 0:
            
            self.product_params['cooldown'][product] -= 1
            # if deviation < np.std(self.product_params['price_history'][product]) :
            #     self.product_params['cooldown'][product] -= self.product_params['cooldown_period'][product] - 1
            #     # return "snap_back", ewma_mid
            return "cooldown", ewma_mid

        # Detect large deviation without trade confirmation (simplified condition)
        if deviation > np.std(self.product_params['price_history'][product])*4.5 and deviation <np.std(self.product_params['price_history'][product])*6.5:
            self.product_params['cooldown'][product] = self.product_params['cooldown_period'][product]
            return "fake", ewma_mid

        return "normal", ewma_mid



    # Trader.calculate_volatility = calculate_volatility
    # Trader.calculate_k = calculate_k
    # Trader.detect_fake = detect_fake
    # Trader.volume_weighted_mid = volume_weighted_mid
    ######### END OF UTILITY FUNCTIONS #########



    ######## UTILITY CODE ######## 
    """ Usefull CodeBlocks"""

    """ Vary Gamma with Calculated Volatility
        
        gamma = max(self.product_params['gamma'][product]/(1+ 20 * realized_vol), 1e-10)

    """

    """Bet against Fake market if detected to loot profit 

        if status == "fake":
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

    """

    """Take all favourable liquidity 
        Basically if there are already orders in favourable prices , take em all

        remaining_buy = self.product_params['max_position'][product] - current_position
        remaining_sell = self.product_params['max_position'][product] + current_position

        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask <= bid_price:
                max_buy = min(self.product_params['max_position'][product] - current_position, -vol)
                if max_buy > 0:
                    orders.append(Order(product, ask, max_buy))
                    remaining_buy -= max_buy
                    current_position += max_buy

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= ask_price:
                max_sell = min(self.product_params['max_position'][product] + current_position, vol)
                if max_sell > 0:
                    orders.append(Order(product, bid, -max_sell))
                    remaining_sell -=max_sell
                    current_position -= max_sell
    """

    """ Multi Level Betting 
        Just place orders from mid price to edge instead of all at bid and ask price
        Increases chances of trade being matched

        num_levels = 3  # Number of price levels
        level_spacing = min(spread , market_spread)/(2*num_levels)
        
        
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

    """

