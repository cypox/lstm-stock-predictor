from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

# Import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# IMPORTANT: need matplotlib==3.2.2 to work, 3.3.2 will raise memory exception


# Create a Stratey
class MLStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('visual', False),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

            if self.params.visual:
                self.show_instance(self.history_cp[0, :, 0], self.predicted_price)

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.history = 30
        self.model = tf.keras.models.load_model("output_model") # TODO: only works with simple models (cant take indicators as input)
        self.model.summary()
        self.predicted_price = -1
        self.history_cp = []

        self.warmup = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def show_instance(self, sequence, prediction, scaler = None):
        data_points = len(sequence)
        plt.figure(figsize=(16, 8))
        plt.plot(sequence, color="blue", label="history")
        plt.plot([data_points-1, data_points], [sequence[-1], prediction], marker='*', markersize=3, color="red", label="prediction")
        plt.show()

    def next(self):
        self.warmup += 1
        if self.warmup < self.history:
            return

        # predict next price
        self.history_cp = np.array([self.dataclose[-i] for i in range(0, self.history)][::-1])
        self.history_cp = np.expand_dims(self.history_cp, 1)
        self.history_cp = np.expand_dims(self.history_cp, 0)
        self.predicted_price = self.model.predict(self.history_cp)[0][0]

        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f, Predicts, %.2f' % (self.dataclose[0], self.predicted_price))

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.predicted_price:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % (self.dataclose[0]))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] > self.predicted_price:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % (self.dataclose[0]))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('Ending Value %.2f' % (self.broker.getvalue()), doprint=True)

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    # strats = cerebro.optstrategy(TestStrategy, maperiod=range(10, 31))
    strats = cerebro.addstrategy(MLStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    ticker = 'TSLA'
    data = bt.feeds.YahooFinanceData(dataname=ticker, fromdate=datetime.datetime(2020, 1, 1), todate=datetime.datetime(2020, 10, 31))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)

    cerebro.plot()
