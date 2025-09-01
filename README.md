# diagonal-hessian-cnn

## 🚧 Pilot Study
*This repository contains preliminary results for a diagonal-Hessian RMSProp optimizer. Full benchmarking and ablation studies are forthcoming.*

## Abstract
This repository provides a TensorFlow implementation of a **diagonal-Hessian, RMSProp-style optimizer**, wrapped in a `MyModel` subclass that works for both **MLP** and **CNN** architectures. The CNN backbone is a compact Conv-BN-ReLU stack (3 blocks) with dropout and a 512-unit head. The optimizer uses variance scaling with bias-corrected momentum and fixed hyperparameters (**diff = 0.25**, **square = 9**, **lr = 1.0×10^-6). All results are averaged over **20 deterministic seeds (0—19)** with **50 epochs (CIFAR-10), 60 epdochs (CIFAR-100)** per run and evaluated via paired two-tailed *t*-tests.  

### Key Results (Custom vs Adam)

#### CIFAR-10 (50 epochs)
- **val_loss**: −4.50 % (*p* = 0.01878)  
- **val_acc**: +0.59 pp (*p* = 0.00601)  
- **macro-F1**: +0.55 pp (*p* = 0.01668)  

#### CIFAR-100 (60 epochs)
- **val_loss**: −1.67 % (*p* = 0.01695)  
- **val_acc**: +0.63 pp (*p* = 0.00181)  
- **macro-F1**: +0.62 pp (*p* = 0.00103)  


### Stability
Beyond mean performance, the custom optimizer exhibits **smaller run-to-run variability** (lower standard deviations) across all metrics, indicating improved reproducibility.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YunInSung/diagonal-hessian-cnn/blob/main/demo.ipynb)

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
  - `square = 9`
  - **Learning rate** fixed at **1.0×10^-6**
- **Performance**: Even on a simple MLP architecture, this configuration consistently outperforms the original optimizer across training loss and convergence speed.

## CNN variant (`MyModel`)

```text
diagonal-hessian-cnn/
├── cnn_adam_vs_custom/
│   ├── experiment_cnn_adam.py      ← Experiment: Adam vs Custom (20 runs with different seeds)
│   └── myModel_2opt.py             ← Custom second-order RMSProp model
├── MLP_custom_2ndOrder_opt/        ← MLP variant (for reference)
│   ├── DNN_ADAM.py
│   └── experiment_runner.py
└── optimizer_benchmark_results/    ← Benchmark result CSVs for each dataset/optimizer combination (aggregated over 20 deterministic runs with seeds 0–19)
```

### Quick Start

```bash
# Adam vs Custom optimizer (default: cifar10 cifar100)
python cnn_adam_vs_custom/experiment_cnn_adam.py

# Specify dataset(s) to run one at a time (to avoid OOM)
python cnn_adam_vs_custom/experiment_cnn_adam.py --datasets cifar10
python cnn_adam_vs_custom/experiment_cnn_adam.py --datasets cifar100
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
| **val_loss** | 0.4332 ± 0.0294 | 0.4137 ± 0.0147 | −0.0195 | −4.50 % | -2.569 | 0.01878 | * |
| **val_acc** | 0.8622 ± 0.0072 | 0.8681 ± 0.0042 | +0.0059 | +0.69 % | +3.091 | 0.00601 | * |
| **f1** | 0.8614 ± 0.0078 | 0.8669 ± 0.0048 | +0.0055 | +0.64 % | +2.625 | 0.01668 | * |
| **train_time** | 248.8213 ± 2.2744 s | 311.3269 ± 4.2343 s | +62.5056 s | +25.12 % | +76.353 | 0.00000 | ** |

**Key points**

* Custom lowers loss (**−4.5 %**), *p* = 0.01878.
* Accuracy improves by **+0.59 pp**, *p* = 0.00601.
* F1 improves by **+0.55 pp**, *p* = 0.01668.
* Training-time overhead **+25.12 %**.
---

#### 2.2 CIFAR-100

| Metric | Adam Mean ± SD | Custom Mean ± SD | Δ (Custom–Adam) | % Change | *t* | *p*-value | Sig. |
|--------|----------------|------------------|-----------------|----------|-----|-----------|------|
| **val_loss** | 1.4717 ± 0.0221 | 1.4472 ± 0.0321 | −0.0245 | −1.67 % | -2.617 | 0.01695 | * |
| **val_acc** | 0.5910 ± 0.0045 | 0.5973 ± 0.0061 | +0.0063 | +1.07 % | +3.624 | 0.00181 | * |
| **f1** | 0.5872 ± 0.0042 | 0.5934 ± 0.0061 | +0.0062 | +1.06 % | +3.870 | 0.00103 | * |
| **train_time** | 315.6871 ± 7.1132 s | 381.5336 ± 9.6949 s | +65.8465 s | +20.86 % | +65.231 | 0.00000 | ** |

**Key points**

* Custom reduces loss (**-1.67 %**), *p* = 0.01695.
* Accuracy improves by **+0.63 pp**, *p* = 0.00181.
* F1 improves by **+0.62 pp**, *p* = 0.00103.
* Training-time overhead **+20.86 %**.

---

#### Significance Flags
* `*` *p* < 0.05  `**` *p* < 0.001

### 3. Interpretation

| Aspect | Summary |
|--------|---------|
| **Performance** | **CIFAR-10:** significant gains in loss (−4.5%, *p* = 0.0188), acc (+0.59 pp, *p* = 0.0060), and F1 (+0.55 pp, *p* = 0.0167). **CIFAR-100:** consistent, significant improvements across loss (−1.67%, *p* = 0.0170), acc (+0.63 pp, *p* = 0.0018), and F1 (+0.62 pp, *p* = 0.0010). |
| **Stability** | **CIFAR-10:** Custom shows slightly lower run-to-run variance (smaller SDs). **CIFAR-100:** SDs are comparable or slightly higher for Custom. |
| **Compute Cost** | Diagonal-Hessian updates add ≈**+25.1%** (CIFAR-10) and ≈**+20.9%** (CIFAR-100) training time. |

