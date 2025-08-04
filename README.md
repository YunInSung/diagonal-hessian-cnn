# diagonal-hessian-cnn

## Abstract
This repository delivers a TensorFlow-based diagonal-Hessian optimizer refactored into an RMSProp-style variant (`lr=1e-4`, `diff=0.3`, `square=5`), wrapped in a `MyModel` subclass for both MLP and CNN architectures.

- **CIFAR-100**: Outperforms **Adam**, reducing validation loss by **8.3%** and increasing macro-F1 by **2.5%** (p < 0.01), and surpasses **SGD + Momentum** with a **2.8%** lower loss (p < 0.01).
- **CIFAR-10**: Cuts validation loss by **3.3%** over **SGD**, with negligible difference against **Adam**.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/YunInSung/diagonal-hessian-cnn/blob/main/demo.ipynb)

## 📋 System Requirements

* **OS**: Ubuntu 22.04 LTS
* **Python**: 3.9
* **CUDA**: 12.1 (nvcc V12.1.105)
* **cuDNN**: 9.9.0
* **TensorFlow**: 2.15.0 (XLA JIT enabled)
* **Main Libraries**

  ```text
  matplotlib==3.9.4
  numpy==1.26.4
  pandas==2.2.3
  scikit-learn==1.6.1
  tensorflow==2.15.0
  tensorflow-addons==0.22.0
  tensorflow-estimator==2.15.0
  tensorflow-io-gcs-filesystem==0.37.1
  tensorflow-probability==0.25.0
  ```

---

## 🛠 Installation

> **GPU vs. CPU**
> A CUDA‑compatible GPU (CUDA 12.1 + cuDNN 9.9) is strongly recommended for reasonable training times. If TensorFlow does not detect a GPU, the scripts automatically fall back to **CPU mode**, which can be **≈ 10× slower** for CIFAR‑10 and substantially more for larger datasets.

```bash
git clone https://github.com/YunInSung/diagonal-hessian-cnn.git
cd diagonal-hessian-cnn

# Create and activate a virtual environment
python3.9 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Modified Custom Optimizer (RMSProp variant)

This repository builds on the original MLP implementation from [relu‑based‑2ndOrder‑convergence](https://github.com/YunInSung/relu-based-2ndOrder-convergence) in two ways:

1. **MLP**: directly forked and patched the original code to incorporate the changes below.  
2. **CNN**: provided a new `tf.keras.Model` subclass (`MyModel`) that embeds the same optimizer logic for convolutional architectures.

All changes applied uniformly to both variants:

- **Removed Hessian‑based dynamic learning rate computed from batch loss** now using a fixed learning rate 
- **Applied standard RMSProp gradient scaling**   
- Hyperparameters adjusted to:
  - `diff = 0.25`
  - `square = 5`
  - **Learning rate** fixed at **0.7×10⁻⁵** for CIFAR-10 and **0.65×10⁻⁵** for CIFAR-100
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

> **Note**: All results are based on 50 epochs of training.

### 1. Experimental Setup
| Item | Details |
|------|---------|
| **Datasets** | CIFAR-10, CIFAR-100 (Keras default train / test splits) |
| **Model** | 3 × {Conv-BN-ReLU-MaxPool-Dropout} → Dense 512 → Softmax |
| **Baselines** | **Adam** (default LR = 0.001, β₁ = 0.9, β₂ = 0.999) |
| **Custom** | `MyModel` (diagonal-Hessian second-order method) – identical network |
| **Runs** | 20 deterministic seeds (0–19), same list for both optimizers |
| **Metrics** | `val_loss`, `val_acc`, macro `f1`, `train_time` |
| **Statistics** | Paired two-tailed *t*-test (α = 0.05); effect size implicit via % change |

---

### 2. Aggregate Results

#### 2.1 CIFAR-10

| Metric | Adam Mean ± SD | Custom Mean ± SD | Δ (Custom–Adam) | % Change | *t* | *p*-value | Sig. |
|--------|----------------|------------------|-----------------|----------|-----|-----------|------|
| **val_loss** | 0.4743 ± 0.0409 | 0.4477 ± 0.0256 | −0.0266 | −5.61 % | −2.734 | 0.01318 | * |
| **val_acc**  | 0.8542 ± 0.0112 | 0.8604 ± 0.0067 | +0.0062 | +0.72 % | +2.301 | 0.03290 | * |
| **f1**       | 0.8540 ± 0.0103 | 0.8593 ± 0.0072 | +0.0053 | +0.62 % | +2.096 | 0.04973 | * |
| **train_time** | 264.7298 ± 2.8424 s | 332.9594 ± 4.0486 s | +68.2296 s | +25.77 % | +94.881 | 0.00000 | ** |

**Key points**

* Custom lowers loss (**−5.6 %**, *p* = 0.013) and raises accuracy/F1 by **+0.62–0.72 pp** (*p* ≈ 0.033/0.050).  
* Training-time overhead **+25.8 %**.

---

#### 2.2 CIFAR-100

| Metric | Adam Mean ± SD | Custom Mean ± SD | Δ (Custom–Adam) | % Change | *t* | *p*-value | Sig. |
|--------|----------------|------------------|-----------------|----------|-----|-----------|------|
| **val_loss** | 1.5666 ± 0.0533 | 1.5198 ± 0.0356 | −0.0467 | −2.98 % | −2.894 | 0.00931 | * |
| **val_acc**  | 0.5700 ± 0.0123 | 0.5786 ± 0.0067 | +0.0086 | +1.51 % | +2.546 | 0.01972 | * |
| **f1**       | 0.5671 ± 0.0121 | 0.5747 ± 0.0068 | +0.0077 | +1.36 % | +2.352 | 0.02960 | * |
| **train_time** | 272.2826 ± 4.3741 s | 341.7940 ± 4.9861 s | +69.5114 s | +25.53 % | +93.681 | 0.00000 | ** |

**Key points**

* Custom **reduces loss (−3.0 %, *p* = 0.009)** and **improves accuracy/F1 by +1.36–1.51 pp** (*p* < 0.03).  
* Training-time penalty is similar to CIFAR-10 (**~+25.5 %**).

---

#### Significance Flags
* `*` *p* < 0.05  `**` *p* < 0.001  

---

### 3. Interpretation

| Aspect | Summary |
|--------|---------|
| **Performance** | *CIFAR-10*: statistically significant gains across loss/acc/F1.  *CIFAR-100*: consistent, significant improvements across all metrics. |
| **Stability** | Custom shows slightly lower run-to-run variance (smaller SDs). |
| **Compute Cost** | Diagonal-Hessian updates add ≈25.8% (CIFAR-10) and ≈25.5% (CIFAR-100) training time. |

---
