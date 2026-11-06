"""Image-based measurement of leaf count and projected area.

A ResNet-18 initialised with ImageNet weights is fine-tuned to regress the
visible leaf count N and the normalised projected area a from a single
top-view RGB frame. Predictions are averaged over the eight dihedral
transformations of the image (test-time augmentation, TTA); the standard
deviation across these transformations is kept as an observable per-frame
uncertainty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dihedral(x, k):
    """k in 0..7: rotation by 90*(k%4) degrees, followed by a flip if k>=4."""
    x = torch.rot90(x, k % 4, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if k >= 4 else x


class Measurer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.features(x))


def to_input(batch_uint8, device):
    x = batch_uint8.to(device, non_blocking=True).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


def train_measurer(images, N, a, train_idx, val_idx, epochs, device, seed=0,
                   batch_size=64, lr=3e-4, log=print):
    """Fine-tune a Measurer. Returns the model (best validation epoch if
    ``val_idx`` is given, otherwise the final epoch) and the per-epoch
    validation history."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    stats = (float(N[train_idx].mean()), float(N[train_idx].std()),
             float(a[train_idx].mean()), float(a[train_idx].std()))
    yN = torch.tensor((N - stats[0]) / stats[1], dtype=torch.float32)
    ya = torch.tensor((a - stats[2]) / stats[3], dtype=torch.float32)

    model = Measurer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    tr = torch.as_tensor(train_idx)
    history, best, best_state = [], np.inf, None
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            x = to_input(images[idx], device)
            x = dihedral(x, int(torch.randint(0, 8, (1,), generator=g)))
            x = x * (1 + 0.1 * (torch.rand(len(idx), 1, 1, 1, device=device) - 0.5))
            target = torch.stack([yN[idx], ya[idx]], 1).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.smooth_l1_loss(model(x).float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if val_idx is not None and len(val_idx):
            pN, pa, _, _, _ = predict(model, images, val_idx, stats, device, tta=1)
            maeN = float(np.mean(np.abs(pN - N[val_idx])))
            maea = float(np.mean(np.abs(pa - a[val_idx])))
            score = maeN / stats[1] + maea / stats[3]
            history.append({"epoch": ep + 1, "val_N_mae": maeN, "val_a_mae": maea})
            log(f"    epoch {ep + 1:2d}  val N-MAE {maeN:.3f}  a-MAE {maea:.4f}")
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.stats = stats
    return model, history


@torch.no_grad()
def predict(model, images, idx, stats, device, tta=8, batch_size=128):
    """Return TTA-mean N, a, their TTA standard deviations and the TTA-mean
    512-d embedding for the frames in ``idx``."""
    model.eval()
    outN, outa, sdN, sda, emb = [], [], [], [], []
    idx = np.asarray(idx)
    for s in range(0, len(idx), batch_size):
        x = to_input(images[idx[s:s + batch_size]], device)
        preds, feats = [], []
        for k in range(tta):
            xk = dihedral(x, k)
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.features(xk).float()
                p = model.head(f).float()
            preds.append(p)
            feats.append(f)
        p = torch.stack(preds)  # (tta, B, 2)
        pN = p[..., 0] * stats[1] + stats[0]
        pa = p[..., 1] * stats[3] + stats[2]
        outN.append(pN.mean(0).cpu())
        outa.append(pa.mean(0).cpu())
        sdN.append(pN.std(0, unbiased=False).cpu())
        sda.append(pa.std(0, unbiased=False).cpu())
        emb.append(torch.stack(feats).mean(0).cpu())
    cat = lambda z: torch.cat(z).numpy()
    return (np.clip(cat(outN), 0, None), np.clip(cat(outa), 0, 1),
            cat(sdN), cat(sda), cat(emb))
