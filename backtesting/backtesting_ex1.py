from backtesting.test import GOOG

GOOG.tail()
GOOG.head()

import pandas as pd

def SMA(values, n):
    return pd.Series(values).rolling(n).mean()


from backtesting import Strategy
from backtesting.lib import crossover

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()        
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

from backtesting import Backtest

bt = Backtest(GOOG, SmaCross, cash=10000, commission=.002)
stats = bt.run()
print(stats)

            
