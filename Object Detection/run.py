import os, json, csv, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import NASFCOSDetector
from controller import NASController, reinforce_update
from dataloaders import COCOFullDataset, COCOValDataset, coco_full_collate, val_collate
from utils import compute_fcos_loss, decode_predictions

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./outputs"
DATA_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)


def get_paths():
    """Data is streamed live from HuggingFace — no local paths needed."""
    return {
        "train_img": None,
        "train_ann": None,
        "val_img":   None,
        "val_ann":   None,
    }



# Phase 1 — NAS Controller Search (FPN + Head)

def evaluate_architecture(fpn_arch, head_arch, share_from, paths, n_steps=150):
    """Quick proxy evaluation: train for n_steps and return negative loss as reward."""
    model = NASFCOSDetector(fpn_arch, head_arch, share_from=share_from).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=8e-4)
    ds    = COCOFullDataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=coco_full_collate)

    model.train()
    total_loss = 0.0
    for step, (imgs, boxes) in enumerate(loader):
        if step >= n_steps:
            break
        imgs  = imgs.to(DEVICE)
        boxes = [b.to(DEVICE) for b in boxes]
        opt.zero_grad()
        cls_p, reg_p, ctr_p = model(imgs)
        loss, _ = compute_fcos_loss(cls_p, reg_p, ctr_p, boxes, 800, 80)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    return -(total_loss / n_steps)   # higher (less negative) = better


def phase1_search(paths, n_archs=5, proxy_steps=150):
    print("\n--- Phase 1: Controller Search (FPN + Head) ---")
    controller = NASController().to(DEVICE)
    opt = torch.optim.Adam(controller.parameters(), lr=3e-3)

    best_reward = -float("inf")
    best_fpn = best_head = best_share = None
    baseline  = None
    rewards   = []

    for step in range(n_archs):
        controller.train()
        fpn_arch, lp_f, ent_f = controller.sample_fpn_arch()
        head_arch, share, lp_h, ent_h = controller.sample_head_arch()

        reward   = evaluate_architecture(fpn_arch, head_arch, share, paths, proxy_steps)
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward

        reinforce_update(lp_f + lp_h, ent_f + ent_h, reward, baseline, opt, controller)
        rewards.append(reward)
        print(f"  Search [{step+1}/{n_archs}] reward={reward:.4f}  baseline={baseline:.4f}")

        if reward > best_reward:
            best_reward = reward
            best_fpn, best_head, best_share = fpn_arch, head_arch, share

    # Save search results
    res = {
        "best_fpn_arch":  best_fpn,
        "best_head_arch": best_head,
        "best_share_from": best_share,
    }
    with open(f"{OUT_DIR}/nas_fcos_search_results.json", "w") as f:
        json.dump(res, f, indent=2)

    plt.figure()
    plt.plot(rewards, marker="o")
    plt.title("NAS Search Rewards")
    plt.xlabel("Architecture #")
    plt.ylabel("Reward (neg. proxy loss)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/nas_fcos_search.png")
    plt.close()

    print(f"Best reward: {best_reward:.4f}")
    return res


# Phase 2 — Full Training on Best Architecture

def full_training(config, paths, n_epochs=12, batch_size=4):
    print("\n--- Phase 2: Full Architecture Training ---")
    model = NASFCOSDetector(
        config["best_fpn_arch"],
        config["best_head_arch"],
        share_from=config["best_share_from"],
    ).to(DEVICE)

    ds     = COCOFullDataset()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=coco_full_collate)

    opt  = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8, 11], gamma=0.1)

    log_path = f"{OUT_DIR}/nas_fcos_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "cls_loss", "reg_loss", "ctr_loss"])

        global_step = 0
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for imgs, boxes in loader:
                imgs  = imgs.to(DEVICE)
                boxes = [b.to(DEVICE) for b in boxes]

                opt.zero_grad()
                cls_p, reg_p, ctr_p = model(imgs, use_checkpoint=True)
                loss, (lc, lr, lct) = compute_fcos_loss(cls_p, reg_p, ctr_p, boxes, 800, 80)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                writer.writerow([epoch, global_step, loss.item(), lc, lr, lct])
                epoch_loss  += loss.item()
                global_step += 1
                n_batches   += 1

                if global_step % 100 == 0:
                    print(f"  Epoch {epoch+1}/{n_epochs}  step {global_step}  "
                          f"loss={loss.item():.4f}  cls={lc:.4f}  reg={lr:.4f}  ctr={lct:.4f}")

            scheduler.step()
            avg = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{n_epochs} done — avg loss: {avg:.4f}  lr: {scheduler.get_last_lr()}")

    torch.save(model.state_dict(), f"{OUT_DIR}/nas_fcos_final.pth")
    print(f"Model saved to {OUT_DIR}/nas_fcos_final.pth")
    return model


# Phase 3 — Evaluation


def evaluation(model, paths):
    print("\n--- Phase 3: Evaluation ---")
    ds     = COCOValDataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=val_collate)

    model.eval()
    results      = []
    first_img_id = None
    first_preds  = []

    with torch.no_grad():
        for imgs, img_ids, scales, orig_ws, orig_hs in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            cls_p, reg_p, ctr_p = model(imgs)

            for i in range(imgs.shape[0]):
                boxes, scores, labels = decode_predictions(
                    [c[i:i+1] for c in cls_p],
                    [r[i:i+1] for r in reg_p],
                    [ct[i:i+1] for ct in ctr_p],
                    800,
                )
                if len(boxes) == 0:
                    continue

                scale  = scales[i]
                boxes  = (boxes.cpu() / scale).tolist()
                scores = scores.cpu().tolist()
                labels = labels.cpu().tolist()

                for b, s, l in zip(boxes, scores, labels):
                    entry = {
                        "image_id":   img_ids[i],
                        "category_id": ds.idx_to_cat[int(l)],
                        "bbox":        [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                        "score":       s,
                    }
                    results.append(entry)
                    if first_img_id is None:
                        first_img_id = img_ids[i]
                    if img_ids[i] == first_img_id:
                        first_preds.append(entry)

    res_file = f"{OUT_DIR}/nas_fcos_val_results.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
    print(f"Evaluation complete. {len(results)} detections written to {res_file}")

    # Visualise first image detections (uses the raw PIL image from the stream)
    if first_preds:
        _visualise_first(first_preds)


def _visualise_first(preds):
    """Re-fetch the first val image from HF stream just for visualisation."""
    try:
        from datasets import load_dataset
        hf_ds   = load_dataset("phiyodr/coco2017", split="validation", streaming=True)
        sample  = next(iter(hf_ds))
        raw_img = sample["image"].convert("RGB")

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(raw_img)
        for r in preds[:10]:
            x, y, w, h = r["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x, y - 4, f"{r['score']:.2f}", color="r", fontsize=8)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/nas_fcos_detections.png")
        plt.close()
        print(f"Detection visualisation saved to {OUT_DIR}/nas_fcos_detections.png")
    except Exception as e:
        print(f"Visualisation skipped: {e}")


# Entry point

if __name__ == "__main__":
    paths  = get_paths()
    config = phase1_search(paths, n_archs=10, proxy_steps=5)
    model  = full_training(config, paths, n_epochs=12, batch_size=4)
    evaluation(model, paths)
