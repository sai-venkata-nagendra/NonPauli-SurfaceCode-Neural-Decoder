# Neural Syndrome Decoder for Non-Pauli Surface-Code Noise

**Author:** Surat Sai Venkata Nagendra Babu, B.Tech CSE, Amrita Vishwa Vidyapeetham

A proof-of-concept study inspired by the work of Prof. Daniel Lidar's group at USC. It uses a simplified phenomenological simulation of a rotated $d=3$ surface code generated in the notebook (not raw trajectories from `QMCtwin` or hardware data from the papers cited below).

## 1. Motivation
In quantum error correction, a decoder reads repeated stabilizer syndrome measurements to infer logical errors. Most shots are error-free while correlated logical errors—especially combined $Y$ flips—are rare, creating severe class imbalance. Having addressed similar multi-class imbalance in my medical-imaging paper (*A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification*, accepted for Springer LNEE, in production), this project tests whether spatiotemporal attention, damped focal loss, and validation threshold calibration transfer to surface-code decoding under non-Pauli noise.

Inspired by:
- **Shen et al. (2026), QMCtwin**, [arXiv:2606.19848](https://arxiv.org/abs/2606.19848): realistic noise ($T_1$ relaxation, $T_2$ dephasing, coherent $ZZ$ crosstalk) produces syndrome statistics that independent Pauli models miss; the authors point toward decoder-facing syndrome models trained on such statistics.
- **Vezvaee et al. (2026)**, [arXiv:2510.18847](https://arxiv.org/abs/2510.18847): surface-code scaling on IBM heavy-hex superconducting processors.

## 2. Simulated Dataset (Generated in Notebook)
- **Code geometry:** Exact rotated $d=3$ surface code (9 data qubits on a $3 \times 3$ grid, 4 $Z$-stabilizers in $H_Z$, and 4 $X$-stabilizers in $H_X$, each with column weight $\le 2$ for exact `PyMatching` compatibility).
- **Scale & splits:** 40,000 shots per random seed (`seeds = [42, 100, 2026]`), 6 syndrome extraction rounds, stratified 70/15/15 train/validation/test split (6,000 test shots per seed).
- **Phenomenological non-Pauli noise model:**
  - Biased $T_1$ bit-flip ($p_X = 0.0080$) and $T_2$ phase-flip ($p_Z = 0.0060$) channels per round.
  - Correlated $Y$ ($X+Z$) errors ($p_Y = 0.0025$) on individual qubits.
  - Coherent spatial $ZZ$ crosstalk ($p_{ZZ} = 0.0055$) across adjacent qubit pairs, with coupled $X$-flip probability ($0.40$).
  - Asymmetric readout (SPAM) noise ($p_{\text{meas}} = 0.0150$, with $1 \to 0$ flips $2.67\times$ more likely than $0 \to 1$) on rounds 1–5.
- **Input:** Consecutive-round syndrome detection events plus total round parity, reshaped to `(6, 3, 3, 1)`.
- **Labels:** Exact logical observable flips (`Logical_I`, `Logical_X`, `Logical_Z`, `Logical_Y`) computed from commutation with $L_Z = [0, 3, 6]$ and $L_X = [0, 1, 2]$.

## 3. Model, Baselines, and Training
**Proposed Architecture (`Focal_CBAM_BiLSTM_Decoder`):**
1. **Two branches per round (`TimeDistributed`):**
   - Branch 1: `Conv2D(48, 3x3)` + `BatchNormalization` + CBAM (`ChannelAttention` + `SpatialAttention`) + `GlobalAveragePooling2D`.
   - Branch 2: `Conv2D(48, 2x2)` + `BatchNormalization` + `Flatten` + `Dense(48)` for local neighbor crosstalk.
2. **Temporal modeling:** Concatenated dual-branch features passed through a `Bidirectional(LSTM(96))` across the 6 rounds, followed by a `Dense(64)` + 4-class softmax head.
3. **Loss & optimization:** Damped Categorical Focal Loss with class weights $\alpha_c \propto (N_{\max} / N_c)^{0.30}$, $\gamma = 1.5$, label smoothing $0.02$, CosineDecay learning rate (`1.5e-3` $\to$ `1e-5`), batch size 128, up to 25 epochs with early stopping (`patience = 5`).
4. **Post-processing:** Per-class probability multipliers tuned on the validation set (`[0.5, 1.5]`) to maximize Macro-F1.

**Baselines:**
- **PyMatching (MWPM):** Static, unweighted independent-Pauli Minimum-Weight Perfect Matching decoded separately on $H_Z$ and $H_X$ using the final-cycle 2D syndrome.
- **MLP (Cross-Entropy):** Flattened static MLP (`Dense(96)` $\to$ `Dropout(0.3)` $\to$ `Dense(48)` $\to$ `Dense(4)`) trained with standard categorical cross-entropy.

## 4. Results (3-Seed Benchmark, 6,000 Test Shots per Seed)

### Summary (Mean ± Std across Seeds 42, 100, 2026)

| Decoder | Logical Error Rate (LER ↓) | Macro F1-Score (↑) | `Logical_Y` Recall (↑) | Micro-Avg ROC-AUC (↑) |
| :--- | :---: | :---: | :---: | :---: |
| **PyMatching (Static Unweighted MWPM)** | `15.10 ± 0.30%` | `69.77 ± 0.37%` | `37.13 ± 1.32%` | N/A (Hard Decision) |
| **Baseline MLP (Cross-Entropy)** | `14.40 ± 0.76%` | `71.45 ± 1.97%` | `43.23 ± 4.98%` | `0.9764 ± 0.0023` |
| **Proposed Focal-CBAM-BiLSTM** | **`10.68 ± 0.99%`** | **`80.46 ± 1.34%`** | **`71.41 ± 1.20%`** | **`0.9822 ± 0.0022`** |

### Per-Seed Breakdown

| Seed | MWPM LER | MLP LER | Proposed LER | MWPM Macro-F1 | MLP Macro-F1 | Proposed Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 14.68% | 13.35% | **9.47%** | 70.18% | 74.14% | **82.12%** |
| **100** | 15.25% | 14.73% | **10.68%** | 69.30% | 70.71% | **80.44%** |
| **2026** | 15.37% | 15.12% | **11.88%** | 69.83% | 69.49% | **78.83%** |

Compared to static unweighted MWPM, the proposed decoder achieves a **29.3% relative reduction in LER** (`15.10%` $\to$ `10.68%`), a **+10.69% gain in Macro-F1**, and nearly **2x higher recall on rare `Logical_Y` errors** (`37.13%` $\to$ `71.41%`) while maintaining `0.93` recall on error-free `Logical_I` shots and class-wise ROC-AUCs above `0.95` across all four classes (`0.967`, `0.975`, `0.957`, `0.967`).

<img width="1326" height="622" alt="image" src="https://github.com/user-attachments/assets/96aed5a5-153b-4e9e-b92b-111d403e6c09" />



## 5. Limitations and Next Steps
This prototype addresses the initial single-seed and cross-entropy-only limitations, but three clear boundaries remain:
1. **Stronger matching baseline:** `PyMatching` here runs static, unweighted 2D matching on the final round rather than a noise-aware 3D spacetime or correlated MWPM decoder.
2. **Code distance scaling:** At $d=3$, the syndrome space per cycle is small ($2^8 = 256$ states); testing whether CBAM + BiLSTM + damped focal loss scales efficiently to $d = 5$ and $d = 7$ is an open next step.
3. **Physics fidelity & ablations:** Replace the phenomenological noise generator with `Stim` detector error models and full Lindblad / `QMCtwin` trajectories, and run separate ablations isolating focal loss vs. architecture on a class-weighted MLP.

## 6. References
1. Harshith, Y. S., Menariya, N., Nagendra Babu, S. S. V., Venkateswaran, S., Batchu, R. K., & Kumar, C. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification.* Springer LNEE, Proceedings of Confluence 2026 (accepted, in production).
2. Shen, T., Chen, H., Huang, B., Takeshita, T., Vezvaee, A., Medalsy, I., & Lidar, D. A. (2026). *QMCtwin: Master-Equation Simulation of Syndrome Statistics Beyond Pauli Noise.* arXiv:2606.19848.
3. Vezvaee, A., Benito, C., Morford-Oberst, M., Bermudez, A., & Lidar, D. A. (2026). *Surface code scaling on heavy-hex superconducting quantum processors.* Nature Communications 17, 9201 (arXiv:2510.18847).
