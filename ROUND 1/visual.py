# backtester_gui.py

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from typing import Dict, List
import importlib.util
import sys

# -- datamodel (custom definition) --
class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class Trade:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class TradingState:
    def __init__(self, timestamp: int, order_depths: Dict[str, OrderDepth], 
                 position: Dict[str, int], ):
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.position = position.copy()
        # self.trader_data = trader_data

# -- backtest engine --
class BacktestResult:
    def __init__(self):
        self.timestamps = []
        self.pnl = []
        self.positions = []

class Backtester:
    PRODUCTS = ["KELP", "RESIN"]
    POSITION_LIMITS = {"KELP": 100, "RESIN": 26}

    def __init__(self):
        self.data: Dict[int, Dict[str, OrderDepth]] = {}
        self.trader_module = None

    def load_data(self, filepath: str):
        self.data.clear()
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = int(row['timestamp'])
                product = row['product']
                if timestamp not in self.data:
                    self.data[timestamp] = {p: OrderDepth() for p in self.PRODUCTS}
                od = self.data[timestamp][product]
                for i in range(1, 4):
                    bid = row.get(f'bid_price_{i}')
                    vol = row.get(f'bid_volume_{i}')
                    if bid and vol:
                        od.buy_orders[int(bid)] = int(vol)
                    ask = row.get(f'ask_price_{i}')
                    vol = row.get(f'ask_volume_{i}')
                    if ask and vol:
                        od.sell_orders[int(ask)] = -int(vol)

    def load_trader(self, filepath: str):
        spec = importlib.util.spec_from_file_location("trader", filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules["trader"] = module
        module.Order = Order
        module.OrderDepth = OrderDepth
        module.Trade = Trade
        module.TradingState = TradingState
        spec.loader.exec_module(module)
        self.trader_module = module

    def run(self):
        if not self.trader_module:
            raise ValueError("Trader module not loaded")
        
        trader = self.trader_module.Trader()
        state_data = ""
        result = BacktestResult()
        pos = {p: 0 for p in self.PRODUCTS}
        pnl = 0

        for ts in sorted(self.data.keys()):
            od = self.data[ts]
            state = TradingState(ts, od, pos.copy(), state_data)
            orders, _, state_data = trader.run(state)
            pnl_delta = 0
            for sym, ol in orders.items():
                for order in ol:
                    depth = od[sym]
                    if order.quantity > 0 and order.price in depth.sell_orders:
                        qty = min(order.quantity, -depth.sell_orders[order.price])
                        pos[sym] += qty
                        pnl_delta -= qty * order.price
                    elif order.quantity < 0 and order.price in depth.buy_orders:
                        qty = min(-order.quantity, depth.buy_orders[order.price])
                        pos[sym] -= qty
                        pnl_delta += qty * order.price
            pnl += pnl_delta
            result.timestamps.append(ts)
            result.pnl.append(pnl)
            result.positions.append(pos.copy())
        return result

# -- GUI --
class BacktestGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prosperity Island Backtester")
        self.geometry("1200x800")
        self.backtester = Backtester()
        self.results = None
        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Data", command=self._load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Load Trader", command=self._load_trader).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Run Backtest", command=self._run).pack(side=tk.LEFT, padx=5)

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.backtester.load_data(path)
            messagebox.showinfo("Success", f"Loaded data from {path}")

    def _load_trader(self):
        path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
        if path:
            self.backtester.load_trader(path)
            messagebox.showinfo("Success", f"Loaded trader from {path}")

    def _run(self):
        try:
            self.results = self.backtester.run()
            self._plot()
        except Exception as e:
            messagebox.showerror("Backtest Failed", str(e))

    def _plot(self):
        if not self.results:
            return
        ts = self.results.timestamps
        pnl = self.results.pnl
        kelp_pos = [p["KELP"] for p in self.results.positions]
        resin_pos = [p["RESIN"] for p in self.results.positions]

        self.axs[0].clear()
        self.axs[0].plot(ts, pnl, label="PnL", color='blue')
        self.axs[0].set_title("PnL Over Time")
        self.axs[0].legend()

        self.axs[1].clear()
        self.axs[1].plot(ts, kelp_pos, label="KELP Pos", color='green')
        self.axs[1].plot(ts, resin_pos, label="RESIN Pos", color='orange')
        self.axs[1].set_title("Positions Over Time")
        self.axs[1].legend()

        self.canvas.draw()

if __name__ == "__main__":
    app = BacktestGUI()
    app.mainloop()
