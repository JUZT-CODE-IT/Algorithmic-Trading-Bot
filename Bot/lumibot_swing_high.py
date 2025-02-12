from config import ALPACA_CONFIG
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader

class SwingHigh(Strategy):
    data = []
    order_number = 0

    def initialize(self):
        self.sleeptime = "10S"

    def on_trading_iteration(self):
        symbol = "GOOG"
        entry_price = self.get_last_price(symbol)
        self.log_message(f"Position: {self.get_position(symbol)}")
        self.data.append(entry_price)
        
        # Log for debugging purposes
        self.log_message(f"Data list: {self.data[-3:]}")

        # Ensure enough data points to make decision
        if len(self.data) > 3:
            temp = self.data[-3:]
            self.log_message(f"Last 3 prints: {temp}")
            
            if temp[-1] > temp[1] > temp[0]:  # Check for uptrend pattern
                order = self.create_order(symbol, quantity=10, side="buy")
                self.submit_order(order)
                self.order_number += 1
                if self.order_number == 1:
                    self.log_message(f"Entry price: {temp[-1]}")
                    entry_price = temp[-1]  # Save the entry price when first order is placed

            # Check if position exists and conditions for exit are met
            if self.get_position(symbol) > 0:
                if self.data[-1] < entry_price * 0.995:  # 0.5% below entry price to sell
                    self.sell_all()
                    self.order_number = 0
                    self.log_message("Sell triggered (below 0.995 entry price).")
                elif self.data[-1] >= entry_price * 1.015:  # 1.5% above entry price to sell
                    self.sell_all()
                    self.order_number = 0
                    self.log_message("Sell triggered (above 1.015 entry price).")

    def before_market_closes(self):
        self.sell_all()  # Ensure positions are cleared before the market closes

if __name__ == "__main__":
    broker = Alpaca(ALPACA_CONFIG)
    strategy = SwingHigh(broker=broker)
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
