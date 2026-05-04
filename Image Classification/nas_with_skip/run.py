# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRS  
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.Pad(padding=4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

class HFCifar10Dataset(Dataset):
    """Wraps a HuggingFace CIFAR-10 split into a PyTorch Dataset."""
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['img']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    batch_size:  int = 128,
    val_size:    int = 5000,
    num_workers: int = 2,
    seed:        int = 42,
    **kwargs,   # absorbs any extra kwargs like data_dir
):
    print("Loading CIFAR-10 from HuggingFace...")
    hf_dataset = load_dataset("uoft-cs/cifar10")

    full_train_aug  = HFCifar10Dataset(hf_dataset['train'], transform=train_transform)
    full_train_eval = HFCifar10Dataset(hf_dataset['train'], transform=eval_transform)
    test_dataset    = HFCifar10Dataset(hf_dataset['test'],  transform=eval_transform)

    train_size = len(full_train_aug) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_aug, [train_size, val_size], generator=gen
    )
    val_subset_eval = Subset(full_train_eval, val_subset.indices)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)

    train_loader = make_loader(train_subset,    shuffle=True)
    val_loader   = make_loader(val_subset_eval, shuffle=False)
    test_loader  = make_loader(test_dataset,    shuffle=False)

    print(f"Dataset split — Train: {len(train_subset):,}  "
          f"Val: {len(val_subset_eval):,}  Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

TOKEN_VALUES = [
    [1, 3, 5, 7],
    [1, 3, 5, 7],
    [1, 2, 3],
    [1, 2, 3],
    [6, 12, 24, 36],
]
TOKENS_PER_LAYER = len(TOKEN_VALUES)


class Controller(nn.Module):
    def __init__(
        self,
        lstm_hidden_size: int = 35,
        lstm_num_layers:  int = 2,
        embedding_dim:    int = 8,
    ):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers
        self.embedding_dim    = embedding_dim

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.softmax_heads = nn.ModuleList([
            nn.Linear(lstm_hidden_size, len(TOKEN_VALUES[i]))
            for i in range(TOKENS_PER_LAYER)
        ])

        self.embeddings = nn.ModuleList([
            nn.Embedding(len(TOKEN_VALUES[i]), embedding_dim)
            for i in range(TOKENS_PER_LAYER)
        ])

        self.start_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.attention_dim = lstm_hidden_size
        self.W_prev = nn.Linear(lstm_hidden_size, self.attention_dim, bias=False)
        self.W_curr = nn.Linear(lstm_hidden_size, self.attention_dim, bias=False)
        self.v      = nn.Linear(self.attention_dim, 1,                bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.08, 0.08)

    def forward(self, num_layers: int = 6):
        device = self.start_token.device

        h0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size, device=device)
        hidden = (h0, c0)

        current_input = self.start_token

        sampled_tokens   = []
        skip_connections = []
        log_probs        = []
        entropies        = []
        anchor_hidden_states = []   # No .detach() — Bug 1 fix

        for layer_idx in range(num_layers):

            for tok_pos in range(TOKENS_PER_LAYER):
                lstm_out, hidden = self.lstm(current_input, hidden)
                h_t = lstm_out[:, -1, :].squeeze(0)

                logits = self.softmax_heads[tok_pos](h_t)
                probs  = F.softmax(logits, dim=-1)

                token_idx = torch.multinomial(probs, num_samples=1).item()
                log_prob  = torch.log(probs[token_idx] + 1e-8)
                entropy   = -(probs * torch.log(probs + 1e-8)).sum()

                sampled_tokens.append(token_idx)
                log_probs.append(log_prob)
                entropies.append(entropy)

                idx_t         = torch.tensor([token_idx], device=device)
                emb           = self.embeddings[tok_pos](idx_t)
                current_input = emb.unsqueeze(0)

            anchor_hidden_states.append(h_t)   # No .detach() — Bug 1 fix

            layer_skips = []

            if layer_idx > 0:
                h_i = anchor_hidden_states[layer_idx]

                for j in range(layer_idx):
                    h_j = anchor_hidden_states[j]

                    score = self.v(
                        torch.tanh(
                            self.W_prev(h_j.unsqueeze(0)) +
                            self.W_curr(h_i.unsqueeze(0))
                        )
                    ).squeeze()

                    prob_connect = torch.sigmoid(score)
                    connect = (torch.rand(1, device=device).item() < prob_connect.item())

                    if connect:
                        layer_skips.append(j)

                    p_val   = prob_connect if connect else (1.0 - prob_connect)
                    log_p   = torch.log(p_val.clamp(min=1e-8))
                    p_c     = prob_connect.clamp(1e-8, 1 - 1e-8)
                    ent_bin = -(p_c * torch.log(p_c) + (1 - p_c) * torch.log(1 - p_c))

                    log_probs.append(log_p)
                    entropies.append(ent_bin)

            skip_connections.append(sorted(layer_skips))

        return sampled_tokens, skip_connections, log_probs, entropies

    def decode(self, tokens: list, num_layers: int) -> list:
        layers = []
        for i in range(num_layers):
            chunk = tokens[i * TOKENS_PER_LAYER : (i + 1) * TOKENS_PER_LAYER]
            layers.append({
                "filter_h":    TOKEN_VALUES[0][chunk[0]],
                "filter_w":    TOKEN_VALUES[1][chunk[1]],
                "stride_h":    TOKEN_VALUES[2][chunk[2]],
                "stride_w":    TOKEN_VALUES[3][chunk[3]],
                "num_filters": TOKEN_VALUES[4][chunk[4]],
            })
        return layers


# ═════════════════════════════════════════════════════════════════════════════
# 3. CHILD NETWORK
# ═════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, filter_h, filter_w, stride_h, stride_w):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(filter_h, filter_w),
            stride=(stride_h, stride_w),
            padding=(filter_h // 2, filter_w // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ChildNetwork(nn.Module):
    def __init__(
        self,
        layer_configs:    list,
        skip_connections: list,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        self.num_layers       = len(layer_configs)
        self.layer_configs    = layer_configs
        self.skip_connections = skip_connections

        self.conv_blocks = nn.ModuleList()
        self._build_layers(in_channels, layer_configs, skip_connections)

        self.gap        = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._head_in_channels, num_classes)

    def _build_layers(self, in_channels, layer_configs, skip_connections):
        dummy      = torch.zeros(1, in_channels, 32, 32)
        feature_maps = []
        consumed   = set()

        for i, cfg in enumerate(layer_configs):
            inputs = []
            if i == 0:
                inputs.append(dummy)
            else:
                inputs.append(feature_maps[i - 1])
                consumed.add(i - 1)

            for j in skip_connections[i]:
                inputs.append(feature_maps[j])
                consumed.add(j)

            if not inputs:
                inputs = [dummy]

            merged = self._merge_inputs(inputs)

            block = ConvBlock(
                merged.shape[1], cfg["num_filters"],
                cfg["filter_h"], cfg["filter_w"],
                cfg["stride_h"], cfg["stride_w"],
            )
            self.conv_blocks.append(block)
            feature_maps.append(block(merged))

        all_ids      = set(range(self.num_layers))
        terminal_ids = sorted(all_ids - consumed)

        terminal_tensors       = [feature_maps[i] for i in terminal_ids]
        merged_terminal        = self._merge_inputs(terminal_tensors)
        self._head_in_channels = merged_terminal.shape[1]
        self._terminal_ids     = terminal_ids

    @staticmethod
    def _merge_inputs(tensors: list) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]

        max_h = max(t.shape[2] for t in tensors)
        max_w = max(t.shape[3] for t in tensors)

        padded = []
        for t in tensors:
            dh = max_h - t.shape[2]
            dw = max_w - t.shape[3]
            if dh > 0 or dw > 0:
                t = F.pad(t, (0, dw, 0, dh))
            padded.append(t)

        return torch.cat(padded, dim=1)

    def forward(self, x):
        x0           = x
        feature_maps = []

        for i, block in enumerate(self.conv_blocks):
            inputs = []
            if i == 0:
                inputs.append(x0)
            else:
                inputs.append(feature_maps[i - 1])

            for j in self.skip_connections[i]:
                inputs.append(feature_maps[j])

            if not inputs:
                inputs = [x0]

            merged = self._merge_inputs(inputs)
            feature_maps.append(block(merged))

        terminal_out = self._merge_inputs([feature_maps[i] for i in self._terminal_ids])
        pooled = self.gap(terminal_out)
        flat   = pooled.view(pooled.size(0), -1)
        return self.classifier(flat)


# ═════════════════════════════════════════════════════════════════════════════
# 4. CHILD TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_child(
    layer_configs:    list,
    skip_connections: list,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    num_epochs:       int = 50,
    device:           torch.device = torch.device("cpu"),
) -> float:
    try:
        model = ChildNetwork(
            layer_configs=layer_configs,
            skip_connections=skip_connections,
        ).to(device)
    except Exception as err:
        print(f"  [WARN] Child build failed: {err} — reward=0")
        return 0.0

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.50), int(num_epochs * 0.75)],
        gamma=0.1,
    )

    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        scheduler.step()

        val_acc = _evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{num_epochs} | "
                  f"val_acc={val_acc:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    best_val = max(val_accuracies[-5:])
    reward   = best_val ** 3
    print(f"    → best val_acc (last 5 epochs): {best_val:.4f} | "
          f"reward: {reward:.6f}")
    return reward


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ═════════════════════════════════════════════════════════════════════════════
# 5. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "lstm_hidden_size":      35,
    "lstm_num_layers":       2,
    "embedding_dim":         8,
    "controller_lr":         0.0006,
    "num_batches":           10,
    "m":                     5,
    "child_epochs":          20, #paper uses 50
    "batch_size":            128,
    "baseline_decay":        0.95,
    "entropy_coeff":         0.0001,
    "start_layers":          6,
    "depth_increase_every":  6,
    "depth_increment":       2,
    "max_layers":            12,
    "seed":                  42,
    "log_dir":               RESULTS_DIR,
}


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN NAS LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_nas():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG["batch_size"],
    )

    controller = Controller(
        lstm_hidden_size=CONFIG["lstm_hidden_size"],
        lstm_num_layers =CONFIG["lstm_num_layers"],
        embedding_dim   =CONFIG["embedding_dim"],
    ).to(device)

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=CONFIG["controller_lr"],
    )

    baseline           = 0.0
    history            = []
    best_reward        = 0.0
    best_val_acc       = 0.0
    best_architecture  = None
    arch_count         = 0
    current_num_layers = CONFIG["start_layers"]

    total_archs = CONFIG["num_batches"] * CONFIG["m"]
    print(f"{'='*65}")
    print(f"  NAS  |  {total_archs} child networks  |  "
          f"{CONFIG['num_batches']} batches × m={CONFIG['m']}")
    print(f"  Start depth: {current_num_layers} layers  |  "
          f"Max depth: {CONFIG['max_layers']} layers")
    print(f"{'='*65}\n")

    for batch_idx in range(CONFIG["num_batches"]):

        print(f"\n{'─'*65}")
        print(f"  Batch {batch_idx+1}/{CONFIG['num_batches']}  |  "
              f"child depth = {current_num_layers} layers")
        print(f"{'─'*65}")

        batch_log_probs   = []
        batch_entropies   = []
        batch_rewards     = []
        batch_layer_cfgs  = []
        batch_skip_cfgs   = []
        batch_tokens_list = []

        for k in range(CONFIG["m"]):
            tokens, skip_conns, log_probs, entropies = controller(
                num_layers=current_num_layers
            )
            layer_configs = controller.decode(tokens, current_num_layers)

            batch_log_probs.append(log_probs)
            batch_entropies.append(entropies)
            batch_layer_cfgs.append(layer_configs)
            batch_skip_cfgs.append(skip_conns)
            batch_tokens_list.append(tokens)

            arch_count += 1
            print(f"\n  [Arch {arch_count:4d} | batch {batch_idx+1}, "
                  f"sample {k+1}/{CONFIG['m']}]")
            for i, (cfg, sk) in enumerate(zip(layer_configs, skip_conns)):
                skip_str = str(sk) if sk else "(none)"
                print(f"    Layer {i}: fh={cfg['filter_h']} fw={cfg['filter_w']} "
                      f"sh={cfg['stride_h']} sw={cfg['stride_w']} "
                      f"nf={cfg['num_filters']}  ← skip from {skip_str}")

        best_updated_this_batch = False   # Bug 3 fix

        for k in range(CONFIG["m"]):
            print(f"\n  Training child {k+1}/{CONFIG['m']} ...")
            reward = train_child(
                layer_configs    = batch_layer_cfgs[k],
                skip_connections = batch_skip_cfgs[k],
                train_loader     = train_loader,
                val_loader       = val_loader,
                num_epochs       = CONFIG["child_epochs"],
                device           = device,
            )
            batch_rewards.append(reward)

            val_acc = reward ** (1 / 3)
            print(f"  Child {k+1}: val_acc={val_acc:.4f}  reward={reward:.6f}")

            if reward > best_reward:
                best_reward    = reward
                best_val_acc   = val_acc
                best_architecture = {
                    "layer_configs":    batch_layer_cfgs[k],
                    "skip_connections": batch_skip_cfgs[k],
                }
                best_updated_this_batch = True
                print(f"  ★ New best val_acc: {best_val_acc:.4f}")

        for r in batch_rewards:
            baseline = (CONFIG["baseline_decay"] * baseline
                        + (1 - CONFIG["baseline_decay"]) * r)

        batch_loss = torch.zeros([], device=device)   # Bug 2 fix

        for k in range(CONFIG["m"]):
            advantage    = batch_rewards[k] - baseline
            log_prob_sum = torch.stack(batch_log_probs[k]).sum()
            entropy_sum  = torch.stack(batch_entropies[k]).sum()

            policy_term  = -log_prob_sum * advantage
            entropy_term = -CONFIG["entropy_coeff"] * entropy_sum
            batch_loss   = batch_loss + policy_term + entropy_term

        batch_loss = batch_loss / CONFIG["m"]

        controller_optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
        controller_optimizer.step()

        # Bug 4 fix: depth schedule on batch boundaries
        completed_batches = batch_idx + 1
        if (completed_batches % CONFIG["depth_increase_every"] == 0
                and current_num_layers < CONFIG["max_layers"]):
            current_num_layers = min(
                current_num_layers + CONFIG["depth_increment"],
                CONFIG["max_layers"],
            )
            print(f"\n  ↑ Depth schedule: num_layers → {current_num_layers}")

        if best_updated_this_batch:   # Bug 3 fix
            _save_checkpoint(
                controller, best_architecture, best_val_acc, arch_count, CONFIG
            )

        mean_reward  = sum(batch_rewards) / CONFIG["m"]
        mean_val_acc = mean_reward ** (1 / 3)

        print(f"\n  Batch {batch_idx+1} summary:")
        print(f"    rewards     : {[f'{r:.4f}' for r in batch_rewards]}")
        print(f"    mean reward : {mean_reward:.6f}  (≈ val_acc {mean_val_acc:.4f})")
        print(f"    baseline    : {baseline:.6f}")
        print(f"    batch_loss  : {batch_loss.item():.6f}")
        print(f"    best so far : val_acc={best_val_acc:.4f}")

        for k in range(CONFIG["m"]):
            history.append({
                "batch_idx":        batch_idx + 1,
                "sample_idx":       k + 1,
                "arch_count":       (batch_idx * CONFIG["m"]) + k + 1,
                "num_layers":       current_num_layers,
                "tokens":           batch_tokens_list[k],
                "layer_configs":    batch_layer_cfgs[k],
                "skip_connections": batch_skip_cfgs[k],
                "reward":           batch_rewards[k],
                "val_acc":          batch_rewards[k] ** (1 / 3),
                "baseline":         baseline,
                "advantage":        batch_rewards[k] - baseline,
            })

    print(f"\n{'='*65}")
    print(f"  NAS complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Best architecture :")
    if best_architecture:
        for i, (cfg, sk) in enumerate(zip(
            best_architecture["layer_configs"],
            best_architecture["skip_connections"],
        )):
            print(f"    Layer {i}: {cfg}  ← skip from {sk if sk else '(none)'}")
    print(f"{'='*65}\n")

    history_path = os.path.join(CONFIG["log_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    return best_architecture, history


# ═════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(controller, best_arch, best_val_acc, arch_idx, config):
    path = os.path.join(config["log_dir"], "best_checkpoint.pt")
    torch.save({
        "arch_idx":          arch_idx,
        "controller_state":  controller.state_dict(),
        "best_architecture": best_arch,
        "best_val_acc":      best_val_acc,
        "config":            config,
    }, path)
    print(f"  Checkpoint saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. FINAL TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_best_on_test(best_architecture, config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"]
    )

    model = ChildNetwork(
        layer_configs    = best_architecture["layer_configs"],
        skip_connections = best_architecture["skip_connections"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True,
    )
    num_epochs = 100
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1,
    )

    print("\nRe-training best architecture for final test evaluation...")
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            val_acc = _evaluate(model, val_loader, device)
            print(f"  epoch {epoch+1}/{num_epochs} | val_acc={val_acc:.4f}")

    test_acc = _evaluate(model, test_loader, device)
    print(f"\n  ★ Final test accuracy : {test_acc:.4f}  "
          f"(error rate: {(1 - test_acc) * 100:.2f}%)")

    # Save test result
    result = {"test_accuracy": test_acc, "error_rate": (1 - test_acc) * 100}
    result_path = os.path.join(RESULTS_DIR, "test_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Test result saved → {result_path}")

    return test_acc


# ═════════════════════════════════════════════════════════════════════════════
# 9. PLOTTING  (saved to results/ for the report)
# ═════════════════════════════════════════════════════════════════════════════

def plot_history(history):
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — no display needed
        import matplotlib.pyplot as plt

        arch_ids  = [h["arch_count"] for h in history]
        val_accs  = [h["val_acc"]    for h in history]
        baselines = [h["baseline"] ** (1/3) for h in history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(arch_ids, val_accs,  label="Val Acc (child)",  alpha=0.7)
        ax.plot(arch_ids, baselines, label="EMA Baseline",     linestyle="--")
        ax.set_xlabel("Architecture index")
        ax.set_ylabel("Validation accuracy")
        ax.set_title("NAS with RL — controller progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(RESULTS_DIR, "nas_progress.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {plot_path}")
    except Exception as e:
        print(f"  [WARN] Plotting failed: {e} (non-fatal)")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAS with RL (Zoph & Le 2017)")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip final test evaluation (faster for debugging)")
    args = parser.parse_args()

    best_arch, history = run_nas()

    plot_history(history)

    if not args.skip_test and best_arch is not None:
        test_acc = evaluate_best_on_test(best_arch)
    else:
        print("\n[INFO] Skipping final test evaluation.")

    print("\nAll outputs saved to:", RESULTS_DIR)
