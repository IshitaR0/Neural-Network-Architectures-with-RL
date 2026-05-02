# data_loader.py
"""
CIFAR-10 data loading with preprocessing pipeline:
    1. Whiten (normalize) all images
    2. Upsample then random 32x32 crop
    3. Random horizontal flip

Split:
    - 45,000 images for training
    - 5,000 images for validation (randomly sampled from the training set)
    - 10,000 images for test (the standard CIFAR-10 test set, touched only once at the end)
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

# CIFAR-10 channel-wise mean and std (pre-computed over the training set).
# These are the standard values used universally for CIFAR-10 whitening.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ── TRAINING TRANSFORM ────────────────────────────────────────────────────────
#
# Following the paper step by step:
#
#   1. Pad by 4 pixels on each side (image goes from 32x32 to 40x40).
#      This is the "upsample" the paper refers to — a cheap way to give
#      the random crop some room to shift around, creating translation
#      augmentation without actually resizing.
#
#   2. RandomCrop(32): cut a random 32x32 patch out of the 40x40 padded image.
#      Combined with step 1, this gives up to ±4 pixel translation in any direction.
#
#   3. RandomHorizontalFlip: flip left-right with 50% probability.
#      CIFAR-10 classes are horizontally symmetric (cars, birds, etc.)
#      so this is a free augmentation.
#
#   4. ToTensor: converts PIL image [H, W, C] uint8 [0,255]
#                to torch Tensor   [C, H, W] float [0.0, 1.0]
#
#   5. Normalize (whiten): subtracts channel mean, divides by channel std.
#      After this, each channel has ~zero mean and ~unit variance.
#      This is what the paper means by "whitening".

train_transform = transforms.Compose([
    transforms.Pad(padding=4),                        # 32x32 -> 40x40
    transforms.RandomCrop(32),                        # random 32x32 patch
    transforms.RandomHorizontalFlip(),                # 50% horizontal flip
    transforms.ToTensor(),                            # to [0,1] float tensor
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD), # whiten
])

# ── VALIDATION / TEST TRANSFORM ───────────────────────────────────────────────
#
# No augmentation at eval time — we want a deterministic, unbiased estimate
# of accuracy. Only whiten (same normalisation as training).

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# GET DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Download CIFAR-10 and return three DataLoaders: train, val, test.

    The paper holds out 5,000 examples from the training set as the
    validation set. The remaining 45,000 are used for training.
    The test set (10,000 examples) is never touched during NAS.

    Parameters
    ----------
    data_dir    : str   where to download/cache the dataset
    batch_size  : int   mini-batch size (128 is standard for CIFAR-10)
    val_size    : int   number of training images to reserve for validation
    num_workers : int   parallel data loading workers (2 is safe for Kaggle)
    seed        : int   for reproducible train/val split

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader, DataLoader, DataLoader
    """

    # ── LOAD RAW TRAINING SET ─────────────────────────────────────────────
    # We load it TWICE with different transforms:
    #   - train_dataset : augmented (for training)
    #   - val_dataset   : not augmented (for validation)
    # Both reference the same underlying 50,000 images but we will split
    # them using the same indices, so they see disjoint subsets.
    #
    # Why load twice? random_split works on indices. If we applied the
    # train_transform to the full dataset and then split, the 5,000
    # validation images would also get augmented — we don't want that.
    # The cleanest fix is two dataset objects sharing the same split indices.

    full_train_aug  = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_train_eval = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform
    )

    # ── SPLIT INTO TRAIN / VAL ────────────────────────────────────────────
    total = len(full_train_aug)           # 50,000
    train_size = total - val_size         # 45,000

    # random_split gives us Subset objects (just index masks, no data copy).
    # We seed the generator so the split is reproducible across runs.
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_aug,
        [train_size, val_size],
        generator=generator,
    )

    # Build the val subset with the eval transform using the SAME indices.
    # Grab the indices that random_split assigned to the val subset.
    val_indices = val_subset.indices
    val_subset_eval = torch.utils.data.Subset(full_train_eval, val_indices)

    # ── TEST SET ──────────────────────────────────────────────────────────
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # ── WRAP IN DATALOADERS ───────────────────────────────────────────────
    #
    # shuffle=True  for training: different order each epoch helps generalisation
    # shuffle=False for val/test: order doesn't matter, just want accuracy
    # pin_memory=True: speeds up CPU->GPU transfers (no-op if using CPU only)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset split:")
    print(f"  Train : {len(train_subset):,} images")
    print(f"  Val   : {len(val_subset_eval):,} images")
    print(f"  Test  : {len(test_dataset):,} images")

    return train_loader, val_loader, test_loader