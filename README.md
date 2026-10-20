# YamAdam
**Ya**mada **m**odified **Adam**

## Abstract
YamAdam is a PyTorch optimizer that requires no user-specified optimizer hyperparameters. It combines moment estimation with historical-only projection, which prevents accumulated history from opposing the current descent direction, and self-calibrated update scaling, which determines the initial update scale from the loss and controls subsequent updates by fixed internal rules. Model architecture, data processing, regularization, batch size, training duration, and stopping criteria remain task-level choices.

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

## Reference
Yamada KD, Adaptive Optimization without User-Specified Optimizer Hyperparameters through Historical-Only Projection and Self-Calibrated Update Scaling, bioRxiv, 2026
