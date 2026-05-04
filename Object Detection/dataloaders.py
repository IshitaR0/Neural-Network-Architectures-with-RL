import torch
import torchvision.transforms.functional as TF
from torch.utils.data import IterableDataset
from datasets import load_dataset

# Standard COCO 80-class category IDs (non-contiguous in COCO spec)
COCO_CAT_IDS = [
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
    22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,
    43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
    62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,
    85,86,87,88,89,90
]
CAT_ID_TO_IDX = {cid: i for i, cid in enumerate(COCO_CAT_IDS)}
IDX_TO_CAT    = {i: cid for i, cid in enumerate(COCO_CAT_IDS)}


def _resize_and_normalize(img, img_size=800):
    w, h = img.size
    scale = img_size / min(w, h)
    if max(w, h) * scale > 1333:
        scale = 1333 / max(w, h)
    img = TF.resize(img, (int(round(h * scale)), int(round(w * scale))))
    img = TF.normalize(TF.to_tensor(img),
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
    return img, scale


class COCOFullDataset(IterableDataset):
    """Streaming COCO train split via HuggingFace — no local download required."""

    def __init__(self, img_dir=None, ann_file=None, img_size=800):
        # img_dir / ann_file kept for API compatibility but unused
        self.img_size = img_size
        self.hf_ds = load_dataset(
            "phiyodr/coco2017",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        for sample in self.hf_ds:
            try:
                img, scale = _resize_and_normalize(
                    sample["image"].convert("RGB"), self.img_size
                )
                objs = sample["objects"]
                boxes = []
                for bbox, cat_id in zip(objs["bbox"], objs["category_id"]):
                    x, y, w_box, h_box = bbox
                    if w_box > 2 and h_box > 2:
                        cls_idx = CAT_ID_TO_IDX.get(cat_id, 0)
                        boxes.append([
                            x * scale, y * scale,
                            (x + w_box) * scale, (y + h_box) * scale,
                            cls_idx
                        ])
                if not boxes:
                    boxes = [[0., 0., 10., 10., 0.]]
                yield img, torch.tensor(boxes, dtype=torch.float32)
            except Exception:
                continue  # skip corrupt / incomplete samples


class COCOValDataset(IterableDataset):
    """Streaming COCO validation split via HuggingFace — no local download required."""

    def __init__(self, img_dir=None, ann_file=None, img_size=800):
        self.img_size = img_size
        self.hf_ds = load_dataset(
            "phiyodr/coco2017",
            split="validation",
            streaming=True,
        )
        self.idx_to_cat = IDX_TO_CAT

    def __iter__(self):
        for sample in self.hf_ds:
            try:
                img, scale = _resize_and_normalize(
                    sample["image"].convert("RGB"), self.img_size
                )
                img_id = sample["image_id"]
                # original size before our resize
                orig_w, orig_h = sample["image"].size
                yield img, img_id, scale, orig_w, orig_h
            except Exception:
                continue


def coco_full_collate(batch):
    import torch.nn.functional as F
    imgs, boxes = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in imgs]
    return torch.stack(padded), list(boxes)


def val_collate(batch):
    import torch.nn.functional as F
    imgs, img_ids, scales, orig_ws, orig_hs = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in imgs]
    return torch.stack(padded), list(img_ids), list(scales), list(orig_ws), list(orig_hs)
