## Modified Custom Optimizer (RMSProp variant)

This repository builds on the original MLP implementation from [relu‑based‑2ndOrder‑convergence](https://github.com/YunInSung/relu-based-2ndOrder-convergence) in two ways:

1. **MLP**: directly forked and patched the original code to incorporate the changes below.  
2. **CNN**: provided a new `tf.keras.Model` subclass (`MyModel`) that embeds the same optimizer logic for convolutional architectures.

All changes applied uniformly to both variants:

- **Removed Hessian‑based dynamic learning rate computed from batch loss** now using a fixed learning rate 
- **Applied standard RMSProp gradient scaling**   
- Hyperparameters adjusted to:
  - `diff = 0.3`
  - `square = 5`
  - **Learning rate** fixed in the range **7.5e‑5 – 1e‑4**
- **Performance**: Even on a simple MLP architecture, this configuration consistently outperforms the original optimizer across training loss and convergence speed.

## CNN variant (`MyModel`)

```text
diagonal-hessian-cnn/
├── cnn_adam_vs_custom/
│   ├── experiment_cnn_adam.py      ← Experiment: Adam vs Custom (20 runs with different seeds)
│   └── myModel_2opt.py             ← Custom second-order RMSProp model
├── cnn_sgd_vs_custom/
│   ├── experiment_cnn_sgd.py       ← Experiment: SGD vs Custom (20 runs with different seeds)
│   └── myModel_2opt.py             ← (same logic, different hyper-parameters)
├── MLP_custom_2ndOrder_opt/        ← MLP variant (for reference)
│   ├── DNN_ADAM.py
│   └── experiment_runner.py
└── optimizer_benchmark_results/    ← Benchmark result CSVs for each dataset/optimizer combination (aggregated over 20 deterministic runs with seeds 0–19)
```

### Quick Start

```bash
# Adam vs Custom optimizer (default: cifar10 cifar100)
python cnn_adam_vs_custom/experiment_cnn_adam.py

# SGD vs Custom optimizer (default: cifar10 cifar100)
python cnn_sgd_vs_custom/experiment_cnn_sgd.py

# Specify dataset(s) to run one at a time (to avoid OOM)
python cnn_adam_vs_custom/experiment_cnn_adam.py --datasets cifar10
python cnn_adam_vs_custom/experiment_cnn_adam.py --datasets cifar100
python cnn_sgd_vs_custom/experiment_cnn_sgd.py --datasets cifar10
python cnn_sgd_vs_custom/experiment_cnn_sgd.py --datasets cifar100
```

Both scripts are self-contained; simply run them to reproduce the experiments.
If you encounter GPU out-of-memory errors, run one dataset at a time using the `--datasets` option or lower the batch size.

### What’s inside `myModel_2opt.py`?

* **Nested `GradientTape`** to capture the gradient *and* the diagonal Hessian in one pass.
* **RMSProp-style variance scaling** combined with bias-corrected first-order momentum.
* A custom `train_step()` that applies a second-order RMSProp update:

  $$
  \theta \leftarrow \theta - \text{lr} \; \frac{\hat m}{\sqrt{\hat v} + \varepsilon}
  $$

The two `myModel_2opt.py` files are identical apart from their default hyper-parameters.

> **Tip**
> Diagonal‑Hessian extraction can be memory‑hungry. If you run into GPU limits, reduce the batch size or enable `jit_compile=True` for XLA acceleration.

## Adam vs Custom Performance Comparison Results

Below table summarizes the mean performance of the Custom optimizer vs. Adam on CIFAR-10/100 datasets (20 deterministic runs with seeds 0–19).

| Dataset       | Metric      | Adam Mean (±std) | Custom Mean (±std) | Mean Diff (Custom–Adam) | % Change | t-stat | p-value      |
| ------------- | ----------- | ---------------- | ------------------ | ----------------------- | -------- | ------ | ------------ |
| **CIFAR-10**  | f1          | 0.7814 ± 0.0260  | 0.8081 ± 0.0097    | +0.0268                 | +3.42%   | 4.305  | 0.00038 \*\* |
|               | val\_acc    | 0.7819 ± 0.0274  | 0.8096 ± 0.0099    | +0.0277                 | +3.54%   | 4.311  | 0.00038 \*\* |
|               | val\_loss   | 0.6624 ± 0.0903  | 0.5920 ± 0.0368    | −0.0704                 | −10.63%  | −3.060 | 0.00644 \*   |
|               | train\_time | 109.46 ± 2.54 s  | 148.28 ± 2.04 s    | +38.82 s                | +35.47%  | 50.689 | \~0 \*\*     |
| **CIFAR-100** | f1          | 0.4212 ± 0.0710  | 0.5049 ± 0.0106    | +0.0837                 | +19.87%  | 5.225  | 0.00005 \*\* |

* **Significance levels**: p < 0.05 (\*), p < 0.001 (\*\*).
* The Custom optimizer achieves statistically significant improvements over Adam on both datasets, but training time increased by \~35.5% on CIFAR-10, with similar overhead on CIFAR-100.

