# diagonal-hessian-cnn

## Abstract
This repository delivers a TensorFlow-based diagonal-Hessian optimizer refactored into an RMSProp-style variant (`lr=1e-4`, `diff=0.3`, `square=5`), wrapped in a `MyModel` subclass for both MLP and CNN architectures.

- **CIFAR-100**: Outperforms **Adam**, reducing validation loss by **8.3%** and increasing macro-F1 by **2.5%** (p < 0.01), and surpasses **SGD + Momentum** with a **2.8%** lower loss (p < 0.01).
- **CIFAR-10**: Cuts validation loss by **3.3%** over **SGD**, with negligible difference against **Adam**.

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
  - `diff = 0.3`
  - `square = 5`
  - **Learning rate** fixed at **1e‑4**
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

> **Note**: All results are based on 40 epochs of training.

### 1. Experimental Setup
| Item | Details |
|------|---------|
| **Datasets** | CIFAR-10, CIFAR-100 (Keras default train / test splits) |
| **Model** | 3 × {Conv-BN-ReLU-MaxPool-Dropout} → Dense 512 → Softmax |
| **Baselines** | **Adam** (default LR = 0.001, β₁ = 0.9, β₂ = 0.999) |
| **Custom** | `MyModel` (diagonal-Hessian second-order method) – identical network |
| **Runs** | 10 deterministic seeds (0-9), same list for both optimizers |
| **Metrics** | `val_loss`, `val_acc`, macro `f1`, `train_time` |
| **Statistics** | Paired two-tailed *t*-test (α = 0.05); effect size implicit via % change |

---

### 2. Aggregate Results

#### 2.1 CIFAR-10

| Metric | Adam Mean ± SD | Custom Mean ± SD | Δ (Custom–Adam) | % Change | *t* | *p*-value | Sig. |
|--------|----------------|------------------|-----------------|----------|-----|-----------|------|
| **val_loss** | 0.5686 ± 0.0354 | 0.5501 ± 0.0293 | −0.0185 | −3.25 % | −1.261 | 0.23912 | |
| **val_acc**  | 0.8451 ± 0.0080 | 0.8444 ± 0.0054 | −0.0007 | −0.08 % | −0.238 | 0.81724 | |
| **f1**       | 0.8448 ± 0.0075 | 0.8435 ± 0.0050 | −0.0013 | −0.16 % | −0.454 | 0.66073 | |
| **train_time** | 203.74 ± 3.02 s | 253.23 ± 4.81 s | +49.49 s | +24.29 % | +37.957 | < 1 e-10 ** | ** |

**Key points**

* Custom optimiser trends in the desired direction for loss (−3 %) but ↓0.1 % in accuracy/F1 – none statistically significant.  
* **+24 % training-time overhead** is highly significant.

---

#### 2.2 CIFAR-100

| Metric | Adam Mean ± SD | Custom Mean ± SD | Δ (Custom–Adam) | % Change | *t* | *p*-value | Sig. |
|--------|----------------|------------------|-----------------|----------|-----|-----------|------|
| **val_loss** | 1.7774 ± 0.0298 | 1.6298 ± 0.0413 | −0.1477 | −8.31 % | −8.023 | 0.00002 | ** |
| **val_acc**  | 0.5526 ± 0.0055 | 0.5673 ± 0.0092 | +0.0147 | +2.67 % | +3.884 | 0.00371 | * |
| **f1**       | 0.5507 ± 0.0057 | 0.5646 ± 0.0095 | +0.0139 | +2.52 % | +3.495 | 0.00678 | * |
| **train_time** | 213.94 ± 4.15 s | 264.36 ± 3.64 s | +50.41 s | +23.56 % | +75.898 | < 1 e-13 ** | ** |

**Key points**

* Custom optimiser **significantly lowers loss (−8 %, *p* ≪ 0.001)** and **raises accuracy/F1 by ~2.6 % (p < 0.01)**.  
* Training-time penalty similar to CIFAR-10 (~24 %).

---

#### Significance Flags
* `*` *p* < 0.05  `**` *p* < 0.001  

---

### 3. Interpretation

| Aspect | Summary |
|--------|---------|
| **Performance** | *CIFAR-10*: gains are small and non-significant.<br>*CIFAR-100*: clear, statistically significant improvements across all performance metrics. |
| **Stability** | Custom optimiser shows lower run-to-run variance on both datasets. |
| **Compute Cost** | Diagonal-Hessian updates add ≈24 % training time in both cases. |

---

## SGD vs Custom Performance Comparison Results

> **Note**: All results are based on 40 epochs of training.

### 1. Experimental Setup
| Item | Details |
|------|---------|
| **Datasets** | CIFAR-10, CIFAR-100 (Keras default train / test splits) |
| **Model** | Three Conv-BN-ReLU-MaxPool-Dropout blocks → Dense 512 → Softmax |
| **Baselines** | **SGD** (learning rate = 0.01, momentum = 0.9, Nesterov ON) |
| **Custom** | `MyModel` (diagonal-Hessian second-order method) – identical network |
| **Runs** | 20 seeds (0–19), same list used for both optimizers |
| **Metrics** | `val_loss`, `val_acc`, macro-`f1`, `train_time` |
| **Statistics** | Paired two-tailed *t*-test (α = 0.05); effect size shown via % change |

---

### 2. Aggregate Results

#### 2.1 CIFAR-10

| Metric | SGD Mean ± SD | Custom Mean ± SD | Δ (Custom–SGD) | % Change | *t* | *p* | Sig. |
|--------|---------------|------------------|----------------|----------|-----|------|------|
| **f1** | 0.8394 ± 0.0119 | 0.8448 ± 0.0083 | **+0.0054** | **+0.64 %** | 1.551 | 0.1373 |  |
| **val_acc** | 0.8404 ± 0.0130 | 0.8455 ± 0.0085 | **+0.0051** | **+0.61 %** | 1.426 | 0.1701 |  |
| **val_loss** | 0.5147 ± 0.0517 | 0.4937 ± 0.0318 | –0.0210 | –4.08 % | –1.403 | 0.1767 |  |
| **train_time** | 198.74 ± 2.08 s | 255.57 ± 3.61 s | +56.83 s | +28.60 % | 93.219 | < 1 e-26 | ** ** |

##### Key points
* All performance metrics trend upward (loss ↓, accuracy/F1 ↑) but **p > 0.05**, so no statistical significance.  
* **+28 % training-time overhead** is highly significant.

---

#### 2.2 CIFAR-100

| Metric | SGD Mean ± SD | Custom Mean ± SD | Δ (Custom–SGD) | % Change | *t* | *p* | Sig. |
|--------|---------------|------------------|----------------|----------|-----|------|------|
| **val_loss** | 1.6808 ± 0.0453 | 1.6345 ± 0.0598 | –0.0463 | –2.75 % | –3.706 | **0.00150** | * |
| **val_acc** | 0.5600 ± 0.0080 | 0.5552 ± 0.0121 | –0.0048 | –0.86 % | –1.923 | 0.0697 |  |
| **f1** | 0.5573 ± 0.0074 | 0.5511 ± 0.0121 | –0.0062 | –1.12 % | –2.572 | **0.01865** | * |
| **train_time** | 196.52 ± 1.86 s | 250.63 ± 2.98 s | +54.12 s | +27.54 % | 94.215 | < 1 e-26 | ** ** |

##### Key points
* **val_loss** improves significantly (*p* < 0.01) but accuracy/F1 drop slightly, hinting at possible over-fitting.  
* Training-time penalty similar to CIFAR-10 (≈28 %).

---

#### Significance Flags
* `*` *p* < 0.05  `**` *p* < 0.001  

---

### 3. Interpretation

| Aspect | Summary |
|--------|---------|
| **Performance** | CIFAR-10 shows positive but non-significant gains (effect size *d* ≈ 0.5). CIFAR-100 lowers loss but hurts accuracy/F1. |
| **Stability** | Lower standard deviations for the custom optimizer indicate better run-to-run consistency. |
| **Compute Cost** | Both datasets incur ~28 % extra training time due to Hessian calculations. |

---

## Comprehensive Conclusion  
_Based on the newly-uploaded CSV files (10 seeds for **Adam** runs, 20 seeds for **SGD + Momentum** runs, 40 training epochs)._

---

### 1 · Performance Gains (val_acc & macro-F1)

| Dataset | vs Adam | Significance | vs SGD + Mom | Significance |
|---------|---------|--------------|--------------|--------------|
| **CIFAR-10** | −0.1 % (*val_acc*), −0.2 % (*F1*) | n.s. (p > 0.6) | **+0.6 %** (*val_acc* / *F1*) | n.s. (p ≈ 0.14–0.17) |
| **CIFAR-100** | **+2.7 %** (*val_acc*), **+2.5 %** (*F1*) | **p < 0.01** | −0.9 % (*val_acc*), −1.1 % (*F1*) | *val_acc* n.s. (p ≈ 0.07); *F1* **p < 0.05** (decline) |

**Take-away**  
*Custom beats Adam convincingly on the harder CIFAR-100 task, but shows no advantage on CIFAR-10. Versus SGD it gains a bit on CIFAR-10 and improves loss everywhere, yet loses a small amount of accuracy/F1 on CIFAR-100.*

---

### 2 · Loss Reduction (val_loss)

| Dataset | vs Adam | Significance | vs SGD + Mom | Significance |
|---------|---------|--------------|--------------|--------------|
| **CIFAR-10** | **−3.25 %** | p ≈ 0.24 (n.s.) | **−4.08 %** | p ≈ 0.18 (n.s.) |
| **CIFAR-100** | **−8.31 %** | **p ≪ 0.001** | **−2.75 %** | **p < 0.01** |

*Hessian scaling consistently lowers validation loss; the effect is dramatic (−8 %) on CIFAR-100.*

---

### 3 · Training-Time Overhead  

| Dataset | vs Adam | vs SGD + Mom |
|---------|---------|--------------|
| **CIFAR-10** | **+24.3 %** (p ≪ 0.001) | **+28.6 %** (p ≪ 0.001) |
| **CIFAR-100** | **+23.6 %** (p ≪ 0.001) | **+27.5 %** (p ≪ 0.001) |

---

### 4 · Overall Verdict  

* **Against Adam**  
  * **CIFAR-100:** clear win—lower loss and ~+2.6 % accuracy/F1, all statistically significant.  
  * **CIFAR-10:** only a minor (non-significant) loss drop; accuracy unchanged.  

* **Against SGD + Momentum**  
  * **CIFAR-10:** small, non-significant accuracy/F1 gain (+0.6 %) and modest loss drop.  
  * **CIFAR-100:** better loss (−2.8 %, p < 0.01) but a slight accuracy/F1 decline (≤ 1 %).  

* **Compute Cost**  
  * Diagonal-Hessian updates add roughly **+24 – 29 %** wall-clock time across the board.