# lstm-stock-predictor

LSTM network to predict stock prices.

Combine indicators and history for better predictions.

# Notes

During phase on, up to the commit this message was added, I was predicting the next stock price based on history. However, the results were not great.

I will change the strategy to predict only if the stock is going up or down. Which should be an easier task.

# Installation

```bash
python3 -m venv env
pip3 install -r requirements.txt
pip3 install https://github.com/cypox/financial-data/archive/master.zip
```

You might also need to change `on_bad_lines` with `error_bad_lines` in `$PYTHON_ENV/lib/python3.8/site-packages/yfinance/utils.py`.

# Running

```bash
source env/bin/activate
python3 src/lsp.py
```

# Output

![alt text](https://github.com/cypox/lstm-stock-predictor/blob/main/output.png?raw=true)

![alt text](https://github.com/cypox/lstm-stock-predictor/blob/main/epoch_loss.svg?raw=true)

Orange: simple network with convolution layers.
Blue: simple network with lstm layers.
Red: double-input network with convolution layers.
Cyan: double-input network with lstm layers.

# TODO

[1] fine tune model
[2] add back-testing
