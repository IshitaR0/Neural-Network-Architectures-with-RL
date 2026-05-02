"""
The child network is the network whose ARCHITECTURE is designed by the controller.
The controller outputs a list of layer configs (filter_h, filter_w, stride_h,
stride_w, num_filters) for each layer. We take those configs and build a real
PyTorch CNN out of them, train it on CIFAR-10, and return its validation accuracy
as the reward signal back to the controller.

Structure of one child network:
    [Conv -> BN -> ReLU] x NUM_LAYERS
    -> Global Average Pooling
    -> Linear(num_filters_last_layer, num_classes)
    -> Softmax (implicit in CrossEntropyLoss)

No skip connections — keeping it simple for now, matching our controller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CONVOLUTIONAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    One building block of the child network:
        Conv2d -> BatchNorm2d -> ReLU

    The controller decides ALL the hyperparameters of this block:
        - kernel size  (filter_h x filter_w)
        - stride       (stride_h x stride_w)
        - num_filters  (= out_channels = depth of the output feature map)

    Parameters
    ----------
    in_channels  : int   number of channels coming IN  (set by previous layer)
    out_channels : int   number of filters / channels going OUT (from controller)
    filter_h     : int   kernel height  (from controller)
    filter_w     : int   kernel width   (from controller)
    stride_h     : int   vertical stride   (from controller)
    stride_w     : int   horizontal stride (from controller)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_h: int,
        filter_w: int,
        stride_h: int,
        stride_w: int,
    ):
        super().__init__()

        # ── PADDING ───────────────────────────────────────────────────────
        # We want the spatial size of the feature map to only shrink because
        # of the STRIDE, not because of the kernel size. The standard trick
        # is "same" padding: pad by floor(kernel_size / 2) on each side.
        # Since our kernels can be non-square, we compute h and w separately.
        #
        # Example: filter_h=5, pad_h=2 → output_height = ceil(H / stride_h)
        pad_h = filter_h // 2
        pad_w = filter_w // 2

        # ── CONV ──────────────────────────────────────────────────────────
        # kernel_size=(filter_h, filter_w) allows non-square kernels, e.g. 1x7.
        # stride=(stride_h, stride_w) similarly.
        # padding=(pad_h, pad_w) keeps spatial dims from shrinking due to kernel.
        # bias=False because BatchNorm right after will absorb any bias term —
        #   BN has its own learnable beta (shift) parameter, so Conv bias is redundant.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(filter_h, filter_w),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            bias=False,
        )

        # ── BATCH NORM ────────────────────────────────────────────────────
        # Normalises each channel across the spatial dimensions of the batch.
        # Helps with training stability — paper explicitly mentions BN.
        # num_features = out_channels (one scale+shift pair per output channel).
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        # x shape: [batch, in_channels, H, W]
        x = self.conv(x)   # -> [batch, out_channels, H', W']
        x = self.bn(x)     # -> same shape, normalised
        x = F.relu(x)      # -> same shape, non-linearity applied element-wise
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CHILD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class ChildNetwork(nn.Module):
    """
    A dynamically-built CNN whose layer configs come entirely from the controller.

    Parameters
    ----------
    layer_configs : list of dicts
        Output of controller.decode(tokens). Each dict looks like:
            {
                'filter_h':    int,
                'filter_w':    int,
                'stride_h':    int,
                'stride_w':    int,
                'num_filters': int,
            }
        The list has one entry per conv layer.

    in_channels : int
        Number of input channels. For CIFAR-10 (RGB images) this is 3.

    num_classes : int
        Number of output classes. For CIFAR-10 this is 10.
    """

    def __init__(
        self,
        layer_configs: list,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        # ── BUILD CONV LAYERS DYNAMICALLY ────────────────────────────────
        # We iterate through the controller's config list and stack ConvBlocks.
        # The key detail: out_channels of layer i = in_channels of layer i+1.
        # We track 'current_channels' for this handoff.
        #
        # nn.ModuleList is used (not a plain Python list) so that PyTorch
        # registers these submodules and includes their parameters in
        # model.parameters() — important for the optimizer to see them.
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels  # starts at 3 for RGB CIFAR-10

        for cfg in layer_configs:
            block = ConvBlock(
                in_channels=current_channels,
                out_channels=cfg["num_filters"],
                filter_h=cfg["filter_h"],
                filter_w=cfg["filter_w"],
                stride_h=cfg["stride_h"],
                stride_w=cfg["stride_w"],
            )
            self.conv_layers.append(block)
            # The next layer's input depth = this layer's output depth
            current_channels = cfg['num_filters']

        # ── GLOBAL AVERAGE POOLING ────────────────────────────────────────
        # After all conv layers, the feature map has some spatial size H' x W'
        # that depends on the strides chosen by the controller. Instead of a
        # fixed-size flattening (which would break if H'/W' varies), we use
        # Global Average Pooling: average each channel's entire spatial map
        # down to a single scalar. Output shape: [batch, last_num_filters, 1, 1].
        #
        # This makes the classifier head always the same size regardless of
        # what strides the controller picked. Very clean.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── CLASSIFIER HEAD ───────────────────────────────────────────────
        # A single linear layer maps from the pooled features to class logits.
        # 'current_channels' at this point = num_filters of the LAST conv layer
        # (whatever the controller decided).
        # num_classes = 10 for CIFAR-10.
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape [batch, 3, 32, 32]   (CIFAR-10 images)

        Returns
        -------
        logits : Tensor, shape [batch, num_classes]
            Raw class scores (not probabilities). CrossEntropyLoss expects these.
        """

        # ── PASS THROUGH CONV STACK ───────────────────────────────────────
        # Each ConvBlock applies Conv -> BN -> ReLU.
        # Spatial size shrinks only where stride > 1.
        for layer in self.conv_layers:
            x = layer(x)
            # After each layer: [batch, num_filters_i, H_i, W_i]

        # ── GLOBAL AVERAGE POOL ───────────────────────────────────────────
        # Collapses H' x W' -> 1 x 1 by averaging each channel.
        # [batch, last_num_filters, H', W'] -> [batch, last_num_filters, 1, 1]
        x = self.global_avg_pool(x)

        # ── FLATTEN ───────────────────────────────────────────────────────
        # Remove the trailing 1 x 1 spatial dims so we get a flat vector.
        # [batch, last_num_filters, 1, 1] -> [batch, last_num_filters]
        x = x.view(x.size(0), -1)

        # ── LINEAR CLASSIFIER ─────────────────────────────────────────────
        # [batch, last_num_filters] -> [batch, num_classes]
        logits = self.classifier(x)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN CHILD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def train_child(
    layer_configs: list,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Build, train, and evaluate one child network. Returns the reward signal
    that will be sent back to the controller.

    Reward (following the paper exactly):
        max validation accuracy over the LAST 5 EPOCHS, cubed.
        Cubing amplifies differences between good and great architectures,
        giving the controller a stronger signal to learn from.

    Parameters
    ----------
    layer_configs : list of dicts   (from controller.decode)
    train_loader  : DataLoader      (45,000 CIFAR-10 training images)
    val_loader    : DataLoader      (5,000 CIFAR-10 validation images)
    num_epochs    : int             (paper uses 50)
    device        : torch.device

    Returns
    -------
    reward : float
        max(val_acc over last 5 epochs) ** 3
        This is a number between 0 and 1.
    """

    # ── BUILD THE CHILD NETWORK ───────────────────────────────────────────
    model = ChildNetwork(layer_configs=layer_configs).to(device)

    # ── LOSS FUNCTION ─────────────────────────────────────────────────────
    # CrossEntropyLoss = LogSoftmax + NLLLoss in one step.
    # Expects raw logits (not softmaxed) as input.
    criterion = nn.CrossEntropyLoss()

    # ── OPTIMIZER (paper spec) ────────────────────────────────────────────
    # SGD with Nesterov Momentum, as specified in the paper:
    #   lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # ── LR SCHEDULE ───────────────────────────────────────────────────────
    #   start at lr=0.1, divide by 10 at 50% and 75% of total epochs.
    #
    # With num_epochs=50:
    #   epochs 0-24  : lr = 0.1
    #   epochs 25-37 : lr = 0.01   
    #   epochs 38-49 : lr = 0.001 
    #
    # MultiStepLR takes a list of epoch milestones and a gamma (multiplicative
    # factor). At each milestone epoch, lr = lr * gamma.
    milestone_50 = int(num_epochs * 0.50)  # e.g. 25
    milestone_75 = int(num_epochs * 0.75)  # e.g. 37
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[milestone_50, milestone_75],
        gamma=0.1,  # divide by 10 at each milestone
    )
    
    # ── TRAINING LOOP ─────────────────────────────────────────────────────
    val_accuracies = []  # we'll collect one per epoch

    for epoch in range(num_epochs):

        # ── TRAIN PHASE ───────────────────────────────────────────────────
        model.train()  # enables dropout, batch norm in train mode

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # clear old gradients
            logits = model(images)             # forward pass
            loss = criterion(logits, labels)   # compute loss
            loss.backward()                    # backprop
            optimizer.step()                   # update weights

        scheduler.step()  # decay the learning rate after each epoch

        # ── VALIDATION PHASE ──────────────────────────────────────────────
        # Only run every epoch (not every batch) — we want the epoch-level acc.
        val_acc = _evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)

        # Print progress every 10 epochs so Kaggle output isn't overwhelming
        if (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch+1:3d}/{num_epochs} | "
                f"val_acc={val_acc:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.5f}"
            )

    # ── COMPUTE REWARD ────────────────────────────────────────────────────
    # Paper: reward = max validation accuracy of the LAST 5 EPOCHS, cubed.
    last_5 = val_accuracies[-5:]
    best_val_acc = max(last_5)
    reward = best_val_acc ** 3

    print(f"  → best val_acc (last 5 epochs): {best_val_acc:.4f}  |  reward: {reward:.6f}")

    return reward, model


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: EVALUATE ON A DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, loader, device) -> float:
    """
    Run model on all batches in loader, return fraction of correct predictions.

    Parameters
    ----------
    model  : nn.Module (child network)
    loader : DataLoader
    device : torch.device

    Returns
    -------
    accuracy : float  (between 0.0 and 1.0)
    """
    model.eval()  # disables dropout / uses running stats for BN
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient computation needed during eval
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            # argmax over class dimension gives predicted class index
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = torch.device("cpu")

    # Simulate what the controller would output after decode()
    dummy_configs = [
        {'filter_h': 3, 'filter_w': 3, 'stride_h': 1, 'stride_w': 1, 'num_filters': 24},
        {'filter_h': 5, 'filter_w': 5, 'stride_h': 1, 'stride_w': 1, 'num_filters': 36},
        {'filter_h': 3, 'filter_w': 3, 'stride_h': 2, 'stride_w': 2, 'num_filters': 48},
        {'filter_h': 1, 'filter_w': 1, 'stride_h': 1, 'stride_w': 1, 'num_filters': 64},
    ]

    model = ChildNetwork(layer_configs=dummy_configs).to(device)

    # Fake a batch of 4 CIFAR-10 images: [batch=4, channels=3, H=32, W=32]
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    logits = model(dummy_input)

    print("ChildNetwork sanity check:")
    print(f"  Input  shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")   # should be [4, 10]
    print(f"  Output (logits):\n{logits}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # Check each conv block's output shape
    print(f"\n  Feature map sizes through the network:")
    x = dummy_input
    for i, layer in enumerate(model.conv_layers):
        x = layer(x)
        print(f"    After conv layer {i}: {tuple(x.shape)}")
    x = model.global_avg_pool(x)
    print(f"    After global avg pool: {tuple(x.shape)}")
    x = x.view(x.size(0), -1)
    print(f"    After flatten:         {tuple(x.shape)}")
