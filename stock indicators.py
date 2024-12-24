import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import logging

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis Dashboard")
        self.root.geometry("1200x800")
        
        self.symbol_var = StringVar(value="TCS.NS")
        self.period_var = StringVar(value="1mo")
        self.current_canvas = None
        self.stock_data = None
        self.setup_logging()
        self.setup_ui()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('StockAnalysis')

    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=X)
        
        ttk.Label(control_frame, text="Stock Symbol:").pack(side=LEFT, padx=5)
        ttk.Entry(control_frame, textvariable=self.symbol_var, width=10).pack(side=LEFT, padx=5)
        
        ttk.Label(control_frame, text="Period:").pack(side=LEFT, padx=5)
        periods = ["1d","1mo", "3mo", "6mo", "1y", "2y", "5y"]
        ttk.Combobox(control_frame, textvariable=self.period_var, values=periods, width=5).pack(side=LEFT, padx=5)
        
        ttk.Button(control_frame, text="Fetch Data", command=self.fetch_and_update).pack(side=LEFT, padx=5)
        
        for indicator in ["Price", "RSI", "MACD", "Bollinger Bands"]:
            ttk.Button(control_frame, text=f"Show {indicator}", 
                      command=lambda i=indicator: self.display_chart(i)).pack(side=LEFT, padx=5)

        # Status label
        self.status_var = StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=RIGHT, padx=5)

    def fetch_stock_data(self):
        try:
            self.status_var.set("Fetching data...")
            self.root.update()
            
            # Validate symbol
            symbol = self.symbol_var.get().strip()
            if not symbol:
                raise ValueError("Please enter a stock symbol")
            
            # Get dates
            end_date = datetime.now()
            period_days = {'1d':40,'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
            start_date = end_date - timedelta(days=period_days[self.period_var.get()])
            
            # Create ticker and verify
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Download data
            self.logger.info(f"Fetching data for {symbol}")
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            if len(data) < 20:  # Minimum required for calculations
                raise ValueError(f"Insufficient data points for {symbol}")
                
            self.status_var.set("Processing data...")
            self.root.update()
            
            processed_data = self.calculate_indicators(data)
            
            if processed_data is not None:
                self.status_var.set("Data ready")
                return processed_data
            
        except ValueError as ve:
            self.logger.error(f"Value Error: {str(ve)}")
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            self.logger.error(f"Fetch Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")
        
        self.status_var.set("Ready")
        return None

    def calculate_indicators(self, data):
        try:
            df = data.copy()
            
            # Basic indicators
            df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_upper'] = df['BB_middle'] + (rolling_std * 2)
            df['BB_lower'] = df['BB_middle'] - (rolling_std * 2)
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Calculation Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to calculate indicators: {str(e)}")
            return None

    def plot_data(self, data, indicator):
        if data is None:
            return None
            
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if indicator == "Price":
                ax.plot(data.index, data['Close'], label='Close Price', color='blue')
                ax.plot(data.index, data['SMA20'], label='SMA20', color='orange')
                ax.plot(data.index, data['SMA50'], label='SMA50', color='red')
                
            elif indicator == "RSI":
                ax.plot(data.index, data['RSI'], label='RSI', color='purple')
                ax.axhline(y=70, color='r', linestyle='--')
                ax.axhline(y=30, color='g', linestyle='--')
                ax.set_ylim([0, 100])
                
            elif indicator == "MACD":
                ax.plot(data.index, data['MACD'], label='MACD', color='blue')
                ax.plot(data.index, data['Signal'], label='Signal', color='red')
                ax.bar(data.index, data['MACD'] - data['Signal'], color='gray', alpha=0.3)
                
            elif indicator == "Bollinger Bands":
                ax.plot(data.index, data['Close'], label='Close', color='blue')
                ax.plot(data.index, data['BB_upper'], label='Upper BB', color='gray')
                ax.plot(data.index, data['BB_lower'], label='Lower BB', color='gray')
                ax.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1)

            ax.set_title(f"{self.symbol_var.get()} - {indicator}")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Plot Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart: {str(e)}")
            return None

    def display_chart(self, indicator):
        if self.stock_data is None:
            messagebox.showwarning("Warning", "Please fetch data first")
            return
            
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            
        fig = self.plot_data(self.stock_data, indicator)
        if fig:
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

    def fetch_and_update(self):
        self.stock_data = self.fetch_stock_data()
        if self.stock_data is not None:
            self.display_chart("Price")

if __name__ == "__main__":
    root = Tk()
    app = StockAnalysisApp(root)
    root.mainloop()
