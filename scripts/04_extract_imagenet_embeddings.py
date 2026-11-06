"""Generic image embeddings from a frozen ImageNet ResNet-18.

These features are not adapted to the leaf-counting task and serve as a
second, task-agnostic representation of the image content for the
visual-increment analysis. Each frame is described by the mean of the global
average-pooled features over the eight dihedral transformations.

Output: outputs/embeddings/imagenet_resnet18.npy  (frames x 512)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.measure import dihedral, to_input  # noqa: E402


@torch.no_grad()
def main():
    C.ensure_dirs()
    device = torch.device("cuda")
    images = torch.from_numpy(np.load(C.CACHE_DIR / "images_160.npy"))
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    out = []
    for s in range(0, len(images), 256):
        x = to_input(images[s:s + 256], device)
        f = torch.stack([net(dihedral(x, k)) for k in range(8)]).mean(0)
        out.append(f.cpu().numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(C.EMB_DIR / "imagenet_resnet18.npy", emb)
    print("embeddings", emb.shape)


if __name__ == "__main__":
    main()
