"""loader mínimo del val limpio de imagenet-100, solo para la
validación end-to-end (sec. 2.4 del spec). reimplementación
autocontenida (val-only) del loader ya usado en el paper 1.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class _RemapEtiquetas(Dataset):
    """envuelve un imagefolder filtrado a los wnids canónicos y
    remapea las etiquetas al orden de la lista (0..99)."""

    def __init__(self, base: datasets.ImageFolder,
                indices: list[int], mapa: dict[int, int]) -> None:
        self.base = base
        self.indices = indices
        self.mapa = mapa

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img, lbl = self.base[self.indices[i]]
        return img, self.mapa[int(lbl)]


def val_loader_imagenet100(
    root: str,
    wnids_path: str,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 8,
) -> DataLoader:
    """loader del split val de imagenet-100, orden canónico cmc.

    args:
        root: carpeta con `val/<wnid>/...` (estructura imagefolder).
        wnids_path: fichero con un synset por línea, orden canónico.
        image_size: lado del crop final.
        batch_size: tamaño de lote.
        num_workers: procesos de carga.

    returns:
        dataloader sin barajar sobre el val completo.
    """
    wnids = [ln.strip() for ln in Path(wnids_path).read_text().splitlines()
             if ln.strip()]
    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        _NORM,
    ])
    base = datasets.ImageFolder(Path(root) / "val", transform=tfm)
    wnid_a_idx = {w: i for i, w in enumerate(wnids)}
    orig_a_nuevo = {
        orig: wnid_a_idx[synset]
        for synset, orig in base.class_to_idx.items()
        if synset in wnid_a_idx
    }
    indices = [i for i, (_, lbl) in enumerate(base.samples)
              if lbl in orig_a_nuevo]
    ds = _RemapEtiquetas(base, indices, orig_a_nuevo)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def top1(modelo: torch.nn.Module, loader: DataLoader,
         disp: str = "cuda") -> float:
    """exactitud top-1 del modelo sobre el loader.

    args:
        modelo: en eval, sobre `disp`.
        loader: dataloader de validación.
        disp: dispositivo.

    returns:
        fracción de aciertos.
    """
    modelo.eval()
    aciertos, total = 0, 0
    for imgs, lbls in loader:
        logits = modelo(imgs.to(disp, non_blocking=True))
        pred = logits.argmax(dim=1).cpu()
        aciertos += int((pred == lbls).sum())
        total += int(lbls.numel())
    return aciertos / total
