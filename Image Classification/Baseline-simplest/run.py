# run.py
from train import run_nas
from child_network import _evaluate, ChildNetwork
from data_loader import get_dataloaders
import torch
import os

if __name__ == "__main__":

    best_arch, history, nas_best_val_acc, device = run_nas()

    ckpt = torch.load(
        os.path.join("./nas_logs/best_checkpoint.pt"),
        map_location=device
    )

    _, _, test_loader = get_dataloaders(batch_size=128, num_workers=0)

    model = ChildNetwork(layer_configs=ckpt["best_architecture"]).to(device)
    model.load_state_dict(ckpt["best_model_state"])

    test_acc = _evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  Final Results")
    print(f"  NAS best val accuracy  : {nas_best_val_acc:.4f}")
    print(f"  Test accuracy          : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}")