# Experimental conditions

This document records the conditions used for the fixed-setting comparison
reported in the YamAdam paper. It describes the workloads, environment,
optimizer settings, and analysis rule.  It does not contain the datasets or
select a favorable setting after observing the results.

## Comparison design

- YamAdam used one unchanged configuration for every workload:
  `YamAdam(model.parameters())`.
- Each comparator setting was treated as a separate method and used unchanged
  on every workload.
- No tuned row, family winner, oracle, best-of-grid result, or
  workload-specific optimizer selection was constructed.
- Seeds were 30, 31, and 32.
- Within each paired workload--seed comparison, both methods used identical
  model initialization, data order, augmentation, numerical precision, and
  update budget.
- Validation was recorded before training and after every optimizer update.
- No test split was used.
- The primary speed coordinate was the number of completed optimizer updates,
  not wall-clock time.
- Every run continued to its fixed update budget, whether or not a target had
  already been reached.

YamAdam contributed 18 trajectories: six workloads by three seeds.  The 30
fixed comparator settings contributed 540 trajectories.  The public comparison
therefore contains 558 unique training runs and 540 paired
setting--workload--seed cases.

## Runtime environment

| Item | Executed condition |
|---|---|
| GPU | NVIDIA Tesla V100-PCIE-16GB |
| Python | 3.12.14 |
| PyTorch | 2.12.1+cu126 |
| CUDA reported by PyTorch | 12.6 |
| NumPy | 2.5.1 |
| Parameter and computation precision | FP32 |
| Automatic mixed precision | disabled |
| Deterministic algorithms | enabled |
| cuDNN deterministic mode | enabled |
| cuDNN benchmark mode | disabled |
| TF32 matrix multiplication | disabled |
| TF32 cuDNN operations | disabled |
| Float32 matrix multiplication precision | `highest` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| PyTorch intra-operation threads | 4 |
| PyTorch inter-operation threads | 1 |

## Workloads

Dataset names below identify the data sources, not canonical training
protocols.  Dataset files are not redistributed in this repository.

### Fashion-MNIST classification

- Data: the 60,000 official training images only.
- Split: fixed class-balanced 48,000/12,000 training/validation split.
- Preprocessing: both partitions use the mean and population standard
  deviation calculated from the training images.
- Model: 3x3 convolutions with channels 1 -> 12 -> 24 -> 48 and strides
  1, 2, 2; GELU activations; then 48x7x7 -> 96 -> 10 fully connected layers.
- Objective: cross-entropy.
- Validation metric: accuracy; larger is better.
- Batch size: 128.
- Budget: five epochs, 1,875 optimizer updates.

### Fashion-MNIST reconstruction

- Data and split: the same fixed 48,000/12,000 split.
- Preprocessing: pixels remain in [0, 1].
- Model: stride-2 convolutional encoder 1 -> 8 -> 16 with GELU; stride-2
  transposed-convolution decoder 16 -> 8 -> 1 with GELU followed by sigmoid.
- Objective and validation metric: pixel mean squared error; smaller is better.
- Batch size: 128.
- Budget: five epochs, 1,875 optimizer updates.

### Adult income

- Data: `adult.data` only.
- Split: identical raw predictor records were kept in the same partition,
  giving 26,049 training and 6,512 validation rows.
- Preprocessing: standardization and one-hot vocabularies were fitted on the
  training partition only, producing 116 inputs.
- Model: GELU multilayer perceptron, 116 -> 64 -> 32 -> 1.
- Objective: binary cross-entropy with logits.
- Validation metric: balanced accuracy; larger is better.
- Batch size: 256.
- Budget: 30 epochs, 3,060 optimizer updates.

### Bike Sharing

- Data: pinned UCI Bike Sharing `hour.csv`.
- Split: first 70% for training, next 15% for validation, final 15% unread.
- Preprocessing: encodings were fitted on the training partition only,
  producing 69 inputs.
- Model: GELU multilayer perceptron, 69 -> 64 -> 32 -> 1.
- Objective: mean squared error on `log(1 + cnt)`.
- Validation metric: log-count root mean squared error; smaller is better.
- Batch size: 256.
- Budget: 84 epochs, 4,032 optimizer updates.

### WikiText-2

- Data: `wiki.train.tokens` and `wiki.valid.tokens` only.
- Vocabulary: two reserved tokens and the 8,192 most frequent training words.
- Model: tied 128-dimensional input/output embeddings and one 128-unit GRU
  layer.
- Objective: token cross-entropy.
- Validation metric: perplexity; smaller is better.
- Batch: 20 sequences by 35 tokens.
- Budget: a seeded sample without replacement of 896 among 2,983 complete
  windows; 896 optimizer updates.

### CIFAR-10

- Data: the five official training batches only.
- Split: fixed class-balanced 45,000/5,000 training/validation split.
- Preprocessing: normalization fitted on the training partition, padded random
  crop, and horizontal flip.
- Model: ResNet-8 with channels 16, 32, and 64, one residual block per stage,
  and group normalization.
- Objective: cross-entropy.
- Validation metric: accuracy; larger is better.
- Batch size: 128.
- Budget: one epoch, 352 optimizer updates.

## Fixed comparator settings

There were three independently reported settings for each comparator family.
Unless stated otherwise, there was no external learning-rate schedule, warmup,
gradient clipping, or output averaging, and the raw optimizer iterate was
evaluated.

### D-Adaptation Adam

- Implementation: official `facebookresearch/dadaptation` `DAdaptAdam`.
- Source commit: `c984980213a989ff7cac1e3d68c19a2257bcd571`.
- Declared version: 3.2.
- Varied `d0`: `1e-8`, `1e-6`, `1e-4`.
- Fixed options: `lr=1.0`, `betas=(0.9, 0.999)`, `eps=1e-8`,
  `weight_decay=0`, `decouple=False`, `use_bias_correction=False`,
  `growth_rate=inf`, `fsdp_in_use=False`, `log_every=0`.

### AdamW

- Implementation: `torch.optim.AdamW` from the executed PyTorch version.
- Varied learning rate: `1e-4`, `1e-3`, `1e-2`.
- Fixed options: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`,
  `amsgrad=False`, `capturable=False`, `differentiable=False`,
  `foreach=False`, `fused=True`, `maximize=False`.

### NAdamW

- Implementation: `torch.optim.NAdam` with decoupled weight decay from the
  executed PyTorch version.
- Varied learning rate: `1e-4`, `1e-3`, `1e-2`.
- Fixed options: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`,
  `momentum_decay=0.004`, `decoupled_weight_decay=True`, `foreach=True`,
  `capturable=False`, `differentiable=False`, `maximize=False`.

### Prodigy

- Implementation: official `konstmish/prodigy` `prodigyopt.Prodigy`.
- Source commit: `3efb213ee8af5a6bf76f28726398433a847b38e9`.
- Varied `d_coef`: `0.5`, `1.0`, `2.0`.
- Fixed options: `lr=1.0`, `betas=(0.9, 0.999)`, `beta3=None`, `eps=1e-8`,
  `weight_decay=0.0`, `d0=1e-6`, `growth_rate=inf`, `decouple=True`,
  `use_bias_correction=False`, `safeguard_warmup=False`, `slice_p=1`,
  `fsdp_in_use=False`.

### Schedule-Free AdamW

- Implementation: official generic `schedulefree.AdamWScheduleFree`, not the
  MLCommons v2 variant.
- Source commit: `70785b53e778d0e872c0bbb75ff4ee54ee10c291`.
- Declared version: 1.4.1.
- Varied learning rate: `1e-4`, `1e-3`, `1e-2`.
- Fixed options: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.0`,
  `warmup_steps=0`, `r=0.0`, `weight_lr_power=2.0`,
  `inner_momentum=0.0`, `foreach=True`.
- The optimizer's internal `x` iterate was reported; the official
  training/evaluation switch was applied around validation.

### Mechanic with Adam

- Implementation: official `optimizedlearning/mechanic`, wrapping
  `torch.optim.Adam`.
- Source commit: `4a037bcd4a9c399167228015ca33a57866d132d9`.
- Declared version: 0.0.1.
- Varied base Adam learning rate: `0.1`, `1.0`, `10.0`.
- Fixed base-Adam options: `betas=(0.9, 0.999)`, `eps=1e-8`,
  `weight_decay=0.0`, `amsgrad=False`, `foreach=False`, `fused=False`,
  `capturable=False`, `differentiable=False`, `maximize=False`.
- Fixed Mechanic options: `s_init=1e-8`, `s_decay=0.01`, `eps=1e-8`,
  `betas=(0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)`,
  `store_delta=False`, `log_every=0`.

### Cautious AdamW

- Implementation: official `kyleliang919/C-Optim` serial AdamW branch.
- Source commit: `a506ee36dbcff48949f1924f31435d1da1a17b72`.
- Varied learning rate: `1e-4`, `1e-3`, `1e-2`.
- Fixed options: `betas=(0.9, 0.999)`, `eps=1e-6`, `weight_decay=0.0`,
  `correct_bias=True`, `foreach=False`, `fused=False`,
  `no_deprecation_warning=True`.

### Grams

- Implementation: official `Gunale0926/Grams` `Grams` class.
- Source commit: `f91a375daaa85c5a4644e3f064ca6693c59f6cdb`.
- Declared version: 0.0.2.
- Varied learning rate: `1e-4`, `1e-3`, `1e-2`.
- Fixed options: `betas=(0.9, 0.999)`, `eps=1e-6`, `weight_decay=0.0`,
  `correct_bias=True`.

### SMORMS3

- Implementation: local PyTorch transcription of the five equations in Simon
  Funk's primary description; no author-code identity is claimed.
- Varied learning-rate cap: `1e-4`, `1e-3`, `1e-2`.
- Fixed option: `eps=1e-16`.

### DoG

- Implementation: official `formll/dog` `DoG` with its official
  `PolynomialDecayAverager`.
- Source commit: `8d45cf569e3824212852d31cc2ae1da7ead34545`.
- Declared version: 1.0.3.
- Varied `reps_rel`: `1e-8`, `1e-6`, `1e-4`.
- Fixed options: `lr=1.0`, `weight_decay=0.0`, `eps=1e-8`,
  `init_eta=None`, averaging `gamma=8.0`.
- The official polynomial-decay averaged iterate was reported.

## Time-to-target analysis

For workload `w`, seed `s`, and progress fraction
`f in {0.05, 0.10, 0.15, 0.20, 0.25}`, let `m0[w,s]` be the common initial
validation metric.  Let `m*[w]` be the mathematical ideal: 1 for accuracy and
perplexity, and 0 for error metrics.  The optimizer-independent target was

```text
target[w,s,f] = m0[w,s] + f * (m*[w] - m0[w,s]).
```

The first completed optimizer update at which the target was reached was
recorded.  An unreached target was assigned the full workload budget for the
restricted-time calculation and was also recorded separately as unreached.

For method `a`, the five target times were divided by the workload budget and
averaged within each workload--seed trajectory.  For comparator setting `c`,
the primary quantity was

```text
Delta_TTT(c) = mean equally over workloads and seeds of
               [restricted_time(c) - restricted_time(YamAdam)].
```

A positive `Delta_TTT` means that YamAdam reached the predefined targets in
fewer optimizer updates on the equal-workload average.

`W / T / L` denotes the number of workload--seed cases in which YamAdam was
faster, tied, or slower after averaging the five targets.  `B / Y / X / N`
denotes target observations reached by both methods, by YamAdam only, by the
comparator only, or by neither method.  The five targets on one trajectory are
nested repeated measurements and are not independent experiments.

## Data availability

The benchmark uses publicly available datasets from their original sources.
No dataset is stored in this repository.  This file records the executed
conditions; `yamadam.py` is the standalone optimizer implementation.
