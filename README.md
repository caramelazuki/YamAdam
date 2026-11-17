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

### Replay scope and limitations

`ReplayableBatchClosure` saves and restores PyTorch's default CPU/CUDA random-number-generator states and the registered buffers of the supplied `model`, including its submodules. This supports replay of modules such as Dropout and BatchNorm. Each search trial starts from the same pre-forward state; when replay finishes, the saved RNG and buffer states from one ordinary base-point forward are restored.

The replay does **not** cover:

- Python's `random`, NumPy random generators, or separately created `torch.Generator` instances.
- Mutable state that is not registered as a buffer of the supplied model, such as ordinary Python attributes, counters, caches, or external state used by the loss function.

The closure may be evaluated more than once during an update. Keep data loading and Python/NumPy-based random preprocessing outside `loss_on_this_batch`, and reuse the resulting tensors. Any other randomness or mutable state that affects the loss must stay fixed during the update or be explicitly replayed by your own closure logic. Otherwise, repeated evaluations can use different randomness or state, making the initial step-size calibration unreliable.

The complete standalone implementation is provided in [`yamadam.py`](yamadam.py). The conditions used for the paper's fixed-setting comparison are recorded in [`EXPERIMENT.md`](EXPERIMENT.md).

## Installation

```bash
python -m pip install "git+https://github.com/caramelazuki/YamAdam.git"
```

## Reference
Yamada KD, [An Adaptive Optimizer without User-Specified Hyperparameters through Historical Projection and Self-Calibrated Scaling](https://www.biorxiv.org/content/10.1101/348557v5), bioRxiv, 2026
