# Neural Architecture Search with Reinforcement Learning

Implementation of [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) (Zoph & Le, 2017).

A controller LSTM learns to generate neural network architectures by treating architecture design as a reinforcement learning problem. 

## Phase 1: 

### Image Classification
1.1 Baseline: The controller samples architectures, trains them on CIFAR-10, and uses validation accuracy as a reward signal to improve future samples via REINFORCE.

---

## Results

- **Best val accuracy (during NAS):** 74.98%
- **Test accuracy:** 74.70%
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
# NAS with skip 

## Overview

The controller learns to propose increasingly better convolutional architectures by:

1. Sampling layer configurations (filter size, stride, number of filters) using an LSTM
2. Sampling skip connections between layers using an attention mechanism
3. Training each sampled ("child") network on CIFAR-10
4. Using validation accuracy as the reward to update the controller via policy gradient

The search progressively increases network depth from 6 to 12 layers over the course of training.

---

## Architecture

```
Controller (LSTM)
    │
    ├─ samples tokens → layer configs (filter_h, filter_w, stride_h, stride_w, num_filters)
    ├─ samples skip connections via attention
    │
    ▼
Child Network (ConvNet)
    │
    ├─ ConvBlocks with BatchNorm + ReLU
    ├─ Skip connections merged by channel-wise concatenation
    ├─ Global Average Pooling
    └─ Linear classifier (10 classes)
    │
    ▼
Validation Accuracy → Reward → Controller Update (REINFORCE)
```

---

## Requirements

```
torch
torchvision
datasets        # HuggingFace datasets
Pillow
matplotlib      # optional, for plotting
```

Install with:

```bash
pip install torch torchvision datasets pillow matplotlib
```

---

## Usage

**Run the full NAS search + final test evaluation:**

```bash
python run.py
```

**Run the search only (skip final test retraining):**

```bash
python run.py --skip-test
```

---

## Configuration

All hyperparameters are set in the `CONFIG` dictionary near the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `lstm_hidden_size` | 35 | Controller LSTM hidden units |
| `lstm_num_layers` | 2 | Controller LSTM depth |
| `embedding_dim` | 8 | Token embedding size |
| `controller_lr` | 0.0006 | Adam LR for controller |
| `num_batches` | 10 | Number of controller update steps |
| `m` | 5 | Child networks sampled per batch |
| `child_epochs` | 20 | Epochs to train each child (paper uses 50) |
| `batch_size` | 128 | Dataloader batch size |
| `baseline_decay` | 0.95 | EMA decay for reward baseline |
| `entropy_coeff` | 0.0001 | Entropy regularisation coefficient |
| `start_layers` | 6 | Initial child network depth |
| `depth_increase_every` | 6 | Batches between depth increments |
| `depth_increment` | 2 | Layers added at each depth increase |
| `max_layers` | 12 | Maximum child network depth |
| `seed` | 42 | Random seed |

---

## Outputs

All outputs are written to the `results/` directory:

| File | Description |
|---|---|
| `results/history.json` | Per-architecture log: tokens, configs, reward, val accuracy, baseline |
| `results/best_checkpoint.pt` | Controller weights + best architecture found |
| `results/nas_progress.png` | Plot of val accuracy and EMA baseline over search |
| `results/test_result.json` | Final test accuracy of the best architecture |

---

## Search Space

Each layer is defined by 5 tokens:

| Token | Values |
|---|---|
| Filter height | 1, 3, 5, 7 |
| Filter width | 1, 3, 5, 7 |
| Stride height | 1, 2, 3 |
| Stride width | 1, 2, 3 |
| Num filters | 6, 12, 24, 36 |

Skip connections between non-adjacent layers are sampled using a learned attention mechanism, allowing the controller to discover residual-style connectivity.

---

## Data

CIFAR-10 is loaded automatically from HuggingFace (`uoft-cs/cifar10`) and split into:

- **Train:** 45,000 images (with augmentation: pad → random crop → horizontal flip)
- **Validation:** 5,000 images
- **Test:** 10,000 images

---

## Results
This is the result after running for 10 baches of 20 epochs each:
<img width="1507" height="427" alt="image" src="https://github.com/user-attachments/assets/4422cf74-07a7-48bd-b38e-4ce7d8d1102d" />

<img width="365" height="38" alt="image" src="https://github.com/user-attachments/assets/c7474a16-8d84-4962-82cc-1553fc436d78" />

<img width="709" height="48" alt="image" src="https://github.com/user-attachments/assets/d6efc532-2407-41b8-8e1f-10c4bc2cb565" />


# NAS-FCOS: Fast Neural Architecture Search for Object Detection

Implementation of [NAS-FCOS: Fast Neural Architecture Search for Object Detection](https://arxiv.org/abs/1906.04423) (Wang et al., 2020).

A controller LSTM learns to generate FPN and detection head architectures for object detection by treating architecture design as a reinforcement learning problem, using FCOS as the base detector.

> **Citation:** Wang, N., Gao, Y., Chen, H., Wang, P., Tian, Z., Shen, C., & Zhang, Y. (2020). NAS-FCOS: Fast neural architecture search for object detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 11943-11951).

---

## Phase 1: FPN Architecture Search

The controller samples FPN block configurations (input nodes, operations, aggregation), trains proxy architectures on COCO, and uses negative proxy loss as a reward signal to improve future samples via REINFORCE.

## Phase 2: Head Architecture Search

The controller samples detection head operation sequences and a weight-sharing split point, again using proxy loss as the reward signal.

## Phase 3: Full Training

The best-found FPN + head architecture is trained end-to-end on COCO train2017 for 12 epochs with SGD and a multi-step LR schedule.

---

## Results

- **Architectures explored:** 10 (proxy-evaluated over 5 steps each)
- **Training epochs:** 12
- **Best FPN architecture:** saved to `outputs/nas_fcos_search_results.json`
- **Final model:** saved to `outputs/nas_fcos_final.pth`
- **Val detections:** saved to `outputs/nas_fcos_val_results.json`

---

## Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point — runs NAS search, full training, and evaluation |
| `controller.py` | LSTM controller that samples FPN and head architectures token by token |
| `models.py` | Backbone (ResNet-50), NAS-FPN blocks, NAS head, and full detector |
| `dataloaders.py` | Streams COCO 2017 from HuggingFace — no local download required |
| `utils.py` | FCOS loss, target generation, and prediction decoding |
| `run.sh` | Sets up environment and launches the pipeline |

---

## Architecture

```
Controller (LSTM)
    │
    ├─ samples FPN block configs (input ids, ops, aggregation)
    ├─ samples head op sequence + weight-sharing split
    │
    ▼
Proxy Detector (NASFCOSDetector)
    │
    ├─ ResNet-50 Backbone → {C3, C4, C5}
    ├─ NAS-FPN (sampled blocks) → {P3, P4, P5, P6, P7}
    └─ NAS Head (sampled ops) → cls / reg / centerness
    │
    ▼
Proxy Loss on COCO → Reward → Controller Update (REINFORCE)
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
psutil>=5.9.0
numpy>=1.23.0
datasets>=2.14.0
```

## Configuration

All hyperparameters are set near the top of `run.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_archs` | 10 | Number of architectures sampled during search |
| `proxy_steps` | 5 | Training steps per proxy evaluation |
| `n_epochs` | 12 | Full training epochs |
| `batch_size` | 4 | Training batch size |
| `controller_lr` | 3e-3 | Adam LR for controller LSTM |
| `train_lr` | 0.01 | SGD LR for full training |
| `momentum` | 0.9 | SGD momentum |
| `weight_decay` | 1e-4 | SGD weight decay |
| `entropy_coeff` | 0.05 | Entropy regularisation coefficient |
| `baseline_decay` | 0.9 | EMA decay for reward baseline |
| `N_FPN_BLOCKS` | 7 | Number of FPN blocks to sample |
| `N_HEAD_OPS` | 6 | Number of head operations to sample |

---

## Search Space

**FPN blocks** — each block is defined by 5 tokens:

| Token | Values |
|-------|--------|
| Input node 1 | Any node in current pool |
| Input node 2 | Any node in current pool |
| Op 1 | sep_conv_3x3, sep_conv_3x3_d3, sep_conv_5x5_d6, skip, deform_3x3 |
| Op 2 | sep_conv_3x3, sep_conv_3x3_d3, sep_conv_5x5_d6, skip, deform_3x3 |
| Aggregation | sum, concat |

**Head ops** — each of the N_HEAD_OPS positions is sampled from:

| Values |
|--------|
| sep_conv_3x3, sep_conv_3x3_d3, sep_conv_5x5_d6, skip, deform_3x3, conv1x1, conv3x3 |

A weight-sharing split point is also sampled, determining how many head ops are level-specific vs. shared across pyramid levels.

---

## Outputs

All outputs are written to the `outputs/` directory:

| File | Description |
|------|-------------|
| `outputs/nas_fcos_search_results.json` | Best FPN arch, head arch, and share_from value |
| `outputs/nas_fcos_search.png` | Plot of search rewards over sampled architectures |
| `outputs/nas_fcos_training_log.csv` | Per-step loss log (epoch, step, total, cls, reg, ctr) |
| `outputs/nas_fcos_final.pth` | Final trained model weights |
| `outputs/nas_fcos_val_results.json` | COCO-format detection results on val2017 |
| `outputs/nas_fcos_detections.png` | Visualisation of detections on the first val image |

---

## Data

COCO 2017 is streamed automatically from HuggingFace (`phiyodr/coco2017`) — no manual download required:

- **Train:** 118,287 images (streamed during search and full training)
- **Validation:** 5,000 images (streamed during evaluation)

---


## How It Works
run `Bash run.sh`

---

