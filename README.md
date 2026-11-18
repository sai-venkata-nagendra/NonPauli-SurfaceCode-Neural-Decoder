# Neural Syndrome Decoder for Imbalanced Syndrome Data (Toy Prototype)

**Author:** Surat Sai Venkata Nagendra Babu
**Affiliation:** B.Tech in Computer Science and Engineering, Amrita Vishwa Vidyapeetham

> **Status: exploratory prototype.** This is a learning project, not a reproduction or extension of any published result. It uses my own toy simulation, not data from the papers below. See [Limitations](#5-limitations).

---

## 1. Motivation

When a quantum computer runs error correction, it repeatedly measures "check" operators (a **syndrome**). A **decoder** reads these syndromes and works out what error really happened. In realistic settings most runs are error-free and the serious logical errors are rare, so the data is **highly imbalanced**.

I had worked on exactly this kind of imbalance in my medical-imaging paper (*A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification*, accepted for Springer LNEE, currently in production). This project tests whether the same ideas (dual-branch networks, attention, focal loss) help on a decoding-style problem.

It was inspired by two papers from Prof. Daniel Lidar's group:
- **Shen et al. (2026), QMCtwin**, [arXiv:2606.19848](https://arxiv.org/abs/2606.19848): a master-equation simulator showing that realistic noise (T1 relaxation, dephasing, ZZ crosstalk) produces syndrome statistics that Pauli models miss. The authors suggest that neural decoders could be trained on such data.
- **Vezvaee et al. (2026)**, [arXiv:2510.18847](https://arxiv.org/abs/2510.18847): surface-code scaling on IBM heavy-hex processors.

This repository does **not** use QMCtwin data or code.

---

## 2. The toy dataset (generated in the notebook)

A simple simulation I wrote myself:
- 25,000 shots, 6 syndrome rounds, 9 data qubits, 8 syndrome checks.
- Each round: random X and Z flips (biased), occasional extra Z flips ("drift"), correlated flips on neighboring qubit pairs, and readout flips that are more likely in one direction.
- Input to the model: the *change* in syndrome between rounds, shaped `(6, 3, 3, 1)`.
- Label: 4 classes (`I`, `X`, `Z`, `Y`) assigned by a simple rule on the final accumulated errors (not by a real decoder).

| Class | Samples | Share |
| :--- | ---: | ---: |
| Logical_I | 20,072 | 80.29% |
| Logical_Z | 3,861 | 15.44% |
| Logical_X | 834 | 3.34% |
| Logical_Y | 233 | 0.93% |

Maximum imbalance: **86.1 : 1**. Split: 70% train / 15% validation / 15% test (stratified, seed 42). The test set has 3,750 shots, including only **35 Y shots**.

---

## 3. Model and training

1. **Two branches per round** (applied with `TimeDistributed`): a 3x3 convolution with channel + spatial attention (CBAM), and a 2x2 convolution for local patterns.
2. **Bidirectional LSTM** across the 6 rounds.
3. **Softmax** over the 4 classes.
4. **Damped focal loss**: class weights proportional to (N_max / N_c)^0.35, gamma = 1.5, label smoothing 0.02. Full inverse-frequency weights over-penalized the common class.
5. Cosine learning-rate decay, early stopping on validation loss.
6. Small class-probability rescaling tuned on validation data.

**Baseline:** a plain MLP (flatten, dense 64, dense 32) with standard cross-entropy and no class weighting.

---

## 4. Results (3,750 test shots, one run, seed 42)

| Class | Baseline recall | Proposed recall | Baseline F1 | Proposed F1 |
| :--- | ---: | ---: | ---: | ---: |
| Logical_I | 96.28% | 85.09% | 89.96% | 89.25% |
| Logical_X | 16.00% | 66.40% | 23.26% | 58.87% |
| Logical_Z | 28.15% | 74.09% | 38.44% | 61.95% |
| Logical_Y | 0.00% | 51.43% (18/35) | 0.00% | 39.13% |
| **Macro F1** | | | **37.92%** | **62.30%** |

Accuracy was essentially unchanged (82.19% vs 82.45%); micro-average ROC-AUC of the proposed model was 0.9721.

**How to read this:** the proposed model catches many more rare errors, but pays for it with more false alarms on error-free shots (recall on `Logical_I` drops from 96% to 85%). The gain is mainly a recall/precision trade-off on rare classes.

![Confusion matrix and ROC curves](SurfaceCode_Neural_Decoder_Results.png)

---

## 5. Limitations

- **Toy noise.** The noise is a simplified Pauli-type model with correlations. It does **not** include non-unital amplitude damping, coherent errors, or master-equation dynamics.
- **Not a full surface-code model.** The checks are a simplified 9-qubit layout, and the labels come from a simple rule, not from running a real decoder.
- **Weak baseline.** The baseline has no class weighting, so part of the gain may come from the focal loss and not the architecture. There is no ablation yet.
- **No standard decoder comparison.** I have not compared against minimum-weight perfect matching (e.g., PyMatching).
- **One seed, small minority class.** All numbers come from one run, and Y recall is 18 of 35 shots, so uncertainty is large.
- The notebook clones Lidar-group repositories for reference but does **not** use their data.

## 6. Planned next steps
1. Rebuild the data with Stim and decode with PyMatching.
2. Add a class-weighted MLP baseline and ablations (loss vs. architecture).
3. Report mean ± std over several random seeds.
4. Try QMCtwin-generated syndrome data if it is available.

## 7. References
1. Harshith, Y. S., Menariya, N., Nagendra Babu, S. S. V., Venkateswaran, S., Batchu, R. K., & Kumar, C. (2026). *A Hybrid Deep Learning Approach for Multi-Class Eye Disease Classification.* Springer LNEE, Confluence 2026 proceedings (accepted, in production).
2. Shen, T., et al., & Lidar, D. A. (2026). *QMCtwin: Master-Equation Simulation of Syndrome Statistics Beyond Pauli Noise.* arXiv:2606.19848.
3. Vezvaee, A., et al., & Lidar, D. A. (2026). *Surface code scaling on heavy-hex superconducting quantum processors.* arXiv:2510.18847.
