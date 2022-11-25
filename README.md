# lstm-stock-predictor

LSTM network to predict stock prices.

Combine indicators and history for better predictions.

# Installation

```bash
python3 -m venv env
pip3 install -r requirements.txt
```

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

[1] database scaling needs to be rewritten (multi-ticker scaler, save/load scaler)
