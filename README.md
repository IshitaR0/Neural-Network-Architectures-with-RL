# Neural Architecture Search with Reinforcement Learning

Implementation of [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) (Zoph & Le, 2017).

A controller LSTM learns to generate neural network architectures by treating architecture design as a reinforcement learning problem. 

## Phase 1: 

### Image Classification
1.1 Baseline: The controller samples architectures, trains them on CIFAR-10, and uses validation accuracy as a reward signal to improve future samples via REINFORCE.

---

## Results

- **Best val accuracy (during NAS):** 74.98%
- **Test accuracy:** 73.40%
- **Architectures explored:** 100 (20 batches × 5 per batch)
- **Best architecture found:**
![Image of the values of best architecture achieved following baseline configurations.](baseline_imageClassification.png)

---

## Files
Image Classification/Baseline-simplest
| File | Purpose |
|------|---------|
| `run.py` | Entry point — runs NAS then evaluates best architecture on test set |
| `train.py` | Outer NAS loop — controller updates via REINFORCE |
| `controller.py` | 2-layer LSTM that samples CNN architectures token by token |
| `child_network.py` | Builds, trains, and evaluates a candidate CNN |
| `data_loader.py` | Loads CIFAR-10 from HuggingFace, handles train/val/test splits |

---

## How It Works
run `Bash run.sh`