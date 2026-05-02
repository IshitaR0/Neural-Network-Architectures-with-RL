# train.py
"""
Main NAS training loop.

The outer loop is the CONTROLLER loop — it runs for many iterations,
each iteration being:
    1. Controller samples one architecture (forward pass)
    2. Child network is built, trained, and evaluated -> reward
    3. REINFORCE update on the controller using the reward

This is the core of Neural Architecture Search.
REINFORCE:
    The controller is a stochastic policy π(a|θ) where:
        a = the architecture (sequence of token choices)
        θ = controller LSTM weights

    We want to maximise E[R] where R is the reward (val accuracy cubed).

    The REINFORCE gradient estimator gives us:
        ∇_θ E[R] ≈ R * Σ_t ∇_θ log π(a_t | a_{<t} ; θ)

    In code: loss = -R * sum(log_probs)
    Then loss.backward() + optimizer.step() does the update.
    The negative sign is because PyTorch minimises loss, but we want to
    MAXIMISE reward.

Baseline:
    Raw REINFORCE has high variance. A standard fix is to subtract a
    baseline b from the reward: (R - b) instead of R.
    If R > b: the architecture was better than average -> increase its probability.
    If R < b: worse than average -> decrease its probability.
    We use an exponential moving average of past rewards as the baseline.
"""

import torch
import os
import json
from datetime import datetime

from controller import Controller, NUM_LAYERS
from child_network import train_child, _evaluate, ChildNetwork
from data_loader import get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Controller
    "lstm_hidden_size": 35,
    "lstm_num_layers": 2,
    "embedding_dim": 8,
    "controller_lr": 0.0006,  # Adam lr from the paper
    # NAS loop
    # The paper samples m architectures per batch, trains all of them,
    # then does ONE controller update using the average gradient.
    # Paper uses m=8 per controller replica. On a single Kaggle GPU,
    # m=5 is a reasonable tradeoff between stability and speed.
    "num_batches": 20,  # how many controller update steps to do
    # total architectures trained = num_batches * m
    "m": 5,  # architectures per batch (paper uses 8)
    # Child network training
    "child_epochs": 50,  # paper uses 50
    "batch_size": 128,
    # REINFORCE baseline
    "baseline_decay": 0.95,  # exponential moving average decay
    # higher = slower to adapt baseline
    # Entropy regularisation
    # Adds a small bonus to the loss proportional to entropy of the
    # controller's distributions. Encourages exploration early in training.
    # Paper doesn't specify a value — 0.0001 is a common small default.
    "entropy_coeff": 0.0001,
    # Misc
    "seed": 42,
    "log_dir": "./nas_logs",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_nas():

    # ── SETUP ─────────────────────────────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    # ── DATA ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG["batch_size"],
    )

    # ── CONTROLLER ────────────────────────────────────────────────────────
    controller = Controller(
        lstm_hidden_size=CONFIG["lstm_hidden_size"],
        lstm_num_layers =CONFIG["lstm_num_layers"],
        embedding_dim   =CONFIG["embedding_dim"],
    ).to(device)

    # Adam optimizer for the controller (paper spec).
    # Only the controller's weights are updated here —
    # each child network has its own separate SGD optimizer inside train_child().
    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=CONFIG["controller_lr"],
    )

    # ── REINFORCE BASELINE ────────────────────────────────────────────────
    # Exponential moving average of rewards.
    # Initialised to 0 - will adapt quickly after the first few architectures.
    baseline = 0.0

    # ── LOGGING ───────────────────────────────────────────────────────────
    history = []   # list of dicts, one per architecture sampled
    best_reward      = 0.0
    best_val_acc     = 0.0
    best_model_state = None
    best_architecture = None

    total_archs = CONFIG["num_batches"] * CONFIG["m"]
    print(f"\n{'='*60}")
    print(f"  Starting NAS")
    print(f"  {CONFIG['num_batches']} batches x m={CONFIG['m']} architectures")
    print(f"  = {total_archs} total child networks to train")
    print(f"{'='*60}\n")

    arch_count = 0  # global counter across all batches

    # ─────────────────────────────────────────────────────────────────────
    # OUTER LOOP: one iteration = one controller update step
    #
    # Each batch:
    #   - Sample m architectures from the controller
    #   - Train all m child networks independently
    #   - Collect their rewards R_1 ... R_m
    #   - Do ONE controller update:
    #       loss = -(1/m) Σ_k  (R_k - b) * Σ_t log P(a_t^k)
    # ─────────────────────────────────────────────────────────────────────
    for batch_idx in range(CONFIG["num_batches"]):

        print(f"\n{'─'*60}")
        print(f"  Batch {batch_idx+1}/{CONFIG['num_batches']}")
        print(f"{'─'*60}")

        # Accumulators for this batch
        batch_log_probs  = []   # list of m items, each is a list of TOTAL_TOKENS log_prob tensors
        batch_entropies  = []   # same structure
        batch_rewards    = []   # list of m scalar floats
        batch_configs    = []   # list of m layer_config lists (for logging)
        batch_tokens     = []   # list of m token lists (for logging)

        # ── STEP 1: SAMPLE m ARCHITECTURES ───────────────────────────────
        # We sample all m architectures BEFORE training any of them.
        # Each sample is an independent forward pass through the controller.
        # Importantly, all m computation graphs are kept alive (log_probs
        # still have gradients) until we call loss.backward() after Step 2.
        for k in range(CONFIG["m"]):
            tokens, log_probs, entropies = controller()
            layer_configs = controller.decode(tokens)

            batch_log_probs.append(log_probs)
            batch_entropies.append(entropies)
            batch_configs.append(layer_configs)
            batch_tokens.append(tokens)

            arch_count += 1
            print(f"\n  [Arch {arch_count:4d} | batch {batch_idx+1}, sample {k+1}/{CONFIG['m']}]")
            print(f"  tokens: {tokens}")
            for i, cfg in enumerate(layer_configs):
                print(f"    Layer {i}: {cfg}")

        # ── STEP 2: TRAIN ALL m CHILD NETWORKS, COLLECT REWARDS ───────────
        # Each child is trained completely independently.
        # We iterate sequentially.
        for k in range(CONFIG["m"]):
            print(f"\n  Training child {k+1}/{CONFIG['m']} ...")
            reward, model = train_child(
                layer_configs=batch_configs[k],
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=CONFIG["child_epochs"],
                device=device,
            )
            batch_rewards.append(reward)

            val_acc = reward ** (1/3)
            print(f"  Child {k+1}: val_acc={val_acc:.4f}  reward={reward:.6f}")

            # Track best architecture seen across all batches
            if reward > best_reward:
                best_reward = reward
                best_val_acc = val_acc
                best_architecture = batch_configs[k]
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ★ New best val_acc: {best_val_acc:.4f}")

        # ── STEP 3: ONE REINFORCE UPDATE USING ALL m REWARDS ──────────────

        # -- 3a. Update the baseline with each reward in the batch --
        # We update the EMA once per reward (sequentially), so the baseline
        # shifts gradually across the batch rather than jumping all at once.
        # This matches the spirit of an EMA baseline in a streaming setting.
        for r in batch_rewards:
            baseline = (
                CONFIG["baseline_decay"] * baseline
                + (1 - CONFIG["baseline_decay"]) * r
            )

        # -- 3b. Compute the batched REINFORCE loss --
        #
        # Paper formula:
        #   ∇J ≈ (1/m) Σ_k  Σ_t  ∇log P(a_t^k) * (R_k - b)
        #
        # As a loss to minimise:
        #   loss = -(1/m) Σ_k  (R_k - b) * Σ_t log P(a_t^k)
        #        + entropy regularisation
        #
        # We build this as a single scalar tensor so one .backward() call
        # accumulates gradients from all m architectures simultaneously.
        batch_loss = torch.tensor(0.0, device=device)

        for k in range(CONFIG["m"]):
            advantage    = batch_rewards[k] - baseline  # scalar float
            log_prob_sum = torch.stack(batch_log_probs[k]).sum()  # scalar tensor
            entropy_sum  = torch.stack(batch_entropies[k]).sum()  # scalar tensor

            # Policy gradient term for this architecture (negated for minimisation)
            policy_term  = -log_prob_sum * advantage

            # Entropy bonus term (subtracted from loss = added to objective)
            entropy_term = -CONFIG["entropy_coeff"] * entropy_sum

            batch_loss = batch_loss + policy_term + entropy_term

        # Average over the batch — (1/m) factor from the paper
        batch_loss = batch_loss / CONFIG["m"]

        # -- 3c. Backprop and update --
        controller_optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping: prevents exploding gradients in the LSTM
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)

        controller_optimizer.step()

        # ── LOGGING ───────────────────────────────────────────────────────
        mean_reward  = sum(batch_rewards) / CONFIG["m"]
        mean_val_acc = mean_reward ** (1/3)

        print(f"\n  Batch {batch_idx+1} summary:")
        print(f"    rewards    : {[f'{r:.4f}' for r in batch_rewards]}")
        print(f"    mean reward: {mean_reward:.6f}  (mean val_acc ≈ {mean_val_acc:.4f})")
        print(f"    baseline   : {baseline:.6f}")
        print(f"    batch_loss : {batch_loss.item():.4f}")
        print(f"    best so far: val_acc={best_val_acc:.4f}")

        for k in range(CONFIG["m"]):
            record = {
                "batch_idx"    : batch_idx + 1,
                "sample_idx"   : k + 1,
                "arch_count"   : (batch_idx * CONFIG["m"]) + k + 1,
                "tokens"       : batch_tokens[k],
                "layer_configs": batch_configs[k],
                "reward"       : batch_rewards[k],
                "val_acc"      : batch_rewards[k] ** (1/3),
                "baseline"     : baseline,
                "advantage"    : batch_rewards[k] - baseline,
            }
            history.append(record)

        # Save checkpoint if this batch contained the best arch so far
        if best_architecture in batch_configs:
            _save_checkpoint(controller, history, CONFIG, arch_count, best_model_state)

    # ─────────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  NAS complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Best architecture :")
    for i, cfg in enumerate(best_architecture):
        print(f"    Layer {i}: {cfg}")
    print(f"{'='*60}\n")

    # Save full history to JSON for analysis
    history_path = os.path.join(CONFIG["log_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Full history saved to {history_path}")

    return best_architecture, history, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(controller, history, config, arch_idx, best_model_state):
    """Save controller weights + best architecture found so far."""
    path = os.path.join(config["log_dir"], "best_checkpoint.pt")
    torch.save({
        "arch_idx"          : arch_idx + 1,
        "controller_state"  : controller.state_dict(),
        "best_architecture" : history[-1]["layer_configs"],
        "best_val_acc"      : history[-1]["val_acc"],
        "best_model_state"  : best_model_state,
        "config"            : config,
    }, path)
    print(f"  Checkpoint saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    best_arch, history, nas_best_val_acc = run_nas()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        batch_size=CONFIG["batch_size"], num_workers=0
    )

    ckpt = torch.load(
        os.path.join(CONFIG["log_dir"], "best_checkpoint.pt"),
        map_location=device
    )

    model = ChildNetwork(layer_configs=ckpt["best_architecture"]).to(device)
    model.load_state_dict(ckpt["best_model_state"])

    test_acc = _evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  Final Results")
    print(f"  NAS best val accuracy  : {nas_best_val_acc:.4f}")
    print(f"  Test accuracy          : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}")