# Spatiotemporal Attention & Focal-Loss Neural Syndrome Decoder for Non-Pauli Surface Codes

**Author:** Surat Sai Venkata Nagendra Babu  
**Affiliation:** B.Tech in Computer Science and Engineering, Amrita Vishwa Vidyapeetham  

---

## 1. Overview & Scientific Motivation

Conventional quantum error-correction (QEC) decoders, such as Minimum Weight Perfect Matching (MWPM), rely on symmetric, independent Pauli noise approximations. However, recent master-equation digital twin simulations and heavy-hex superconducting processor experiments by Professor Daniel A. Lidar's group at the USC Center for Quantum Information Science & Technology (CQIST)—specifically **Shen et al. ([arXiv:2606.19848](https://arxiv.org/abs/2606.19848))** and **Vezvaee et al. ([Nature Communications 2026 / arXiv:2510.18847](https://arxiv.org/abs/2510.18847))**—demonstrate that realistic superconducting hardware exhibits **non-unital amplitude damping ($T_1$ relaxation bias)**, **coherent $ZZ$ crosstalk**, and **asymmetric readout (SPAM) errors**.

From a machine learning perspective, decoding multi-cycle syndrome measurements under non-unital noise presents two critical challenges:
1. **Spatiotemporal Correlations:** Coherent crosstalk between coupled neighboring qubits and temporal drift across measurement cycles violate static independence assumptions.
2. **Extreme Class Imbalance:** In sub-threshold NISQ operating regimes, the trivial logical outcome (`Logical_I`) dominates the syndrome distribution (>80%), while combined bit-and-phase logical flips (`Logical_Y`) represent less than 1% of shots (an **86.1:1 imbalance ratio**). Standard Cross-Entropy neural decoders collapse toward the majority class, resulting in near-zero recall on rare logical errors.

This repository extends the **Dual-Backbone Feature Fusion**, **CBAM (Channel & Spatial Attention)**, **Bidirectional LSTM**, and **Categorical Focal Loss with Label Smoothing** framework from our peer-reviewed medical imaging research (*"A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification"*, accepted in **Springer Nature LNEE**, 2026) to build a proof-of-concept **Spatiotemporal Neural Syndrome Decoder** for $d=3$ surface codes.

---

## 2. Dataset & Phenomenological Master-Equation Noise Model

The simulation generates $N = 25,000$ shots across $T = 6$ syndrome extraction cycles on a 9-data-qubit ($3 \times 3$), 8-ancilla-qubit $d=3$ surface code (`shape: (25000, 6, 3, 3, 1)`), incorporating:
* **Biased $T_1 / T_2$ Relaxation:** Physical bit-flip ($p_X = 0.0085$) and phase-flip ($p_Z = 0.0045$) channels with non-Markovian temporal carry-over.
* **Spatial Coherent Crosstalk:** Coupled neighbor $ZZ / XX$ interactions ($p_{ZZ} = 0.0035$) across adjacent lattice edges.
* **Asymmetric Readout Errors:** Biased measurement assignment errors ($p_{\text{meas}} = 0.0120$, with $1 \to 0$ relaxation occurring $2.67\times$ more frequently than $0 \to 1$ excitation).

### Class Distribution (86.1:1 Maximum Imbalance)
* **`Logical_I (Safe)`:** 20,072 samples (80.29%)
* **`Logical_Z (PhaseFlip)`:** 3,861 samples (15.44%)
* **`Logical_X (BitFlip)`:** 834 samples (3.34%)
* **`Logical_Y (Combined)`:** 233 samples (0.93%)

The dataset is partitioned via stratified sampling into **70% Training (17,500)**, **15% Validation (3,750)**, and **15% Testing (3,750)**.

---

## 3. Proposed Neural Decoder Architecture

1. **Dual-Branch Spatiotemporal Feature Extractor (`TimeDistributed`):**
   * **Branch 1 (Global Stabilizer Attention):** `Conv2D(32, 3x3)` + `BatchNormalization` + **Channel Attention** + **Spatial Attention (CBAM)** + `GlobalAveragePooling2D` to capture full-grid syndrome patterns at each cycle.
   * **Branch 2 (Local Crosstalk Extractor):** `Conv2D(32, 2x2)` + `BatchNormalization` + `Dense(32)` to isolate nearest-neighbor two-qubit crosstalk correlations.
2. **Temporal Sequence Modeling:** Concatenated dual-branch features (`64-dim` per cycle) are processed by a **Bidirectional LSTM (`64 units`)** to model forward and backward error propagation chains across $t = 1 \dots 6$.
3. **Damped Categorical Focal Loss & Prior Threshold Calibration:**
   $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \alpha_c \propto \left(\frac{N_{\max}}{N_c}\right)^{0.35}$$
   Trained with $\gamma = 1.5$, label smoothing $\epsilon = 0.02$, Cosine Annealing learning rate schedule, and validation-set threshold calibration (`[1.0, 0.9, 1.05, 1.0]`).

---

## 4. Experimental Results (3,750 Unseen Test Shots)

| Logical Class | Baseline Precision | Baseline Recall | Baseline F1-Score | Proposed Precision | Proposed Recall | Proposed F1-Score | Proposed ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Logical_I (Safe)`** | 84.42% | 96.28% | 89.96% | **93.85%** | 85.09% | 89.25% | **0.900** |
| **`Logical_X (BitFlip)`** | 42.55% | 16.00% | 23.26% | **52.87%** | **66.40%** | **58.87%** | **0.965** |
| **`Logical_Z (PhaseFlip)`** | 60.59% | 28.15% | 38.44% | **53.23%** | **74.09%** | **61.95%** | **0.911** |
| **`Logical_Y (Combined)`** | 0.00% | 0.00% | 0.00% | **31.58%** | **51.43%** | **39.13%** | **0.862** |
| **Overall / Macro Avg** | 46.89% | 35.11% | **37.92%** | **57.88%** | **69.25%** | **62.30% (+24.38%)** | **0.9721 (Micro)** |

![Confusion Matrix and ROC Curves](Lidar_SurfaceCode_Neural_Decoder_Results.png)

### Key Takeaways
* **+24.38% Macro F1-Score Improvement:** The proposed architecture lifts Macro F1 from **37.92%** (Standard Cross-Entropy MLP decoder) to **62.30%** while improving overall test accuracy (`82.19%` $\to$ `82.45%`).
* **Eliminates Minority-Class Collapse:** Whereas the standard Cross-Entropy baseline completely fails to identify rare `Logical_Y` errors (`0.00%` recall), our Damped Focal-Loss + CBAM + BiLSTM decoder achieves **51.43% recall** on `Logical_Y`, **66.40% recall** on `Logical_X` (vs. `16.00%`), and **74.09% recall** on `Logical_Z` (vs. `28.15%`), with a **0.9721 micro-average ROC-AUC**.

---

## 5. References
1. Shen, T., Chen, H., Huang, B., Takeshita, T., Vezvaee, A., Medalsy, I., & Lidar, D. A. (2026). *A Master-Equation Digital Twin for Quantum Error-Correction Syndrome Statistics Beyond Pauli Noise*. arXiv:2606.19848. GitHub: `USCqserver/DataRepo-Surface-Code-QMC-Sim`.
2. Vezvaee, A., et al., & Lidar, D. A. (2026). *Surface code scaling on heavy-hex superconducting quantum processors*. Nature Communications 17, 9201 (arXiv:2510.18847).
3. Harshith, Y. S., Menariya, N., Nagendra Babu, S. S. V., Venkateswaran, S., Batchu, R. K., & Kumar, C. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification*. Springer Nature Lecture Notes in Electrical Engineering (LNEE).
