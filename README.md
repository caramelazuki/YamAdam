# YamAdam
YamAdam (**Ya**mada **m**odified **Adam**) is a PyTorch optimizer that requires no user-specified optimizer hyperparameters. It combines moment estimation with historical-only projection, which prevents accumulated history from opposing the current descent direction, and self-calibrated update scaling, which determines the initial update scale from the loss and controls subsequent updates by fixed internal rules.

## Usage
```python
from yamadam import ReplayableBatchClosure, YamAdam

optimizer = YamAdam(model.parameters())

model.train()
for inputs, targets in train_loader:
    def loss_on_this_batch():
        predictions = model(inputs)
        return criterion(predictions, targets)

    closure = ReplayableBatchClosure(model, loss_on_this_batch)
    loss = optimizer.step(closure)
```

YamAdam computes the required gradients inside `optimizer.step`. Do not call `optimizer.zero_grad()` or `loss.backward()` separately. The closure must evaluate the same selected batch whenever it is called during an update.

The complete standalone implementation is provided in [`yamadam.py`](yamadam.py). The conditions used for the paper's fixed-setting comparison are recorded in [`EXPERIMENT.md`](EXPERIMENT.md).

## Installation

```bash
python -m pip install "git+https://github.com/caramelazuki/YamAdam.git"
```

## Reference
Yamada KD, [An Adaptive Optimizer without User-Specified Hyperparameters through Historical Projection and Self-Calibrated Scaling](https://www.biorxiv.org/content/10.1101/348557v5), bioRxiv, 2026
