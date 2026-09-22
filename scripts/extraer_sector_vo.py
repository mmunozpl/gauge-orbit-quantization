"""extrae el sector valor-salida a un portador ligero para la demo.

la demo no puede cargar el vit de timm ni pythia entero, y tampoco
debe reimplementar el gauge: reimplementarlo perdería la propiedad
que hace fiable esta ruta, que la demo ejecuta el mismo código que
produjo las tablas del paper. el portador es la solución mínima: los
tensores $(W_v, W_O)$ por cabeza, en la convención de este repo
(w_v [d_h, d], w_o [d, d_h]), y nada más.

se guardan en float32, que es la dtype con la que el modelo los
lleva. `src.weights.w_v_w_o_*` hace `.double()` sobre esos mismos
float32, así que subir el portador a float64 al cargarlo devuelve
tensores idénticos bit a bit a los que el barrido consumió. la
aritmética de gauges sigue siendo fp64, como manda el proyecto.

uso:
    python scripts/extraer_sector_vo.py --modelo vitb
    python scripts/extraer_sector_vo.py --modelo pythia
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.weights import (cargar_pythia, cargar_vitb, w_v_w_o_pythia,
                         w_v_w_o_vitb)

SALIDA_DIR = "artifacts/demo"
MODELOS = {
    "pythia": {"n_capas": 24, "n_cabezas": 16, "dim_cabeza": 64},
    "vitb": {"n_capas": 12, "n_cabezas": 12, "dim_cabeza": 64},
}


def _carga(nombre: str, cfg: dict):
    """carga el modelo y su función de extracción por-cabeza."""
    if nombre == "pythia":
        return cargar_pythia(cfg["pythia_model_id"]), w_v_w_o_pythia
    return cargar_vitb(cfg["vitb_ckpt"]), w_v_w_o_vitb


def sha256(ruta: Path) -> str:
    """digest del fichero, para el manifiesto.

    args:
        ruta: fichero a resumir.

    returns:
        sha256 en hexadecimal.
    """
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    """extrae el sector de un modelo y escribe portador y manifiesto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, choices=list(MODELOS))
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = MODELOS[args.modelo]

    print(f"[extraer] cargando {args.modelo}...", flush=True)
    modelo, extrae = _carga(args.modelo, cfg)

    w_v_capas, w_o_capas = [], []
    for capa in tqdm(range(info["n_capas"]), desc=args.modelo):
        w_v, w_o = extrae(modelo, capa, info["n_cabezas"],
                          info["dim_cabeza"])
        # se vuelve a float32: es la dtype de origen, y el .double()
        # de la extracción no añadió información.
        w_v_capas.append(w_v.float())
        w_o_capas.append(w_o.float())
    tensores = {"w_v": torch.stack(w_v_capas),
                "w_o": torch.stack(w_o_capas)}

    out = Path(SALIDA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fichero = out / f"sector_vo_{args.modelo}.safetensors"
    save_file(tensores, str(fichero))

    manifiesto = {
        "modelo": args.modelo,
        "n_capas": info["n_capas"],
        "n_cabezas": info["n_cabezas"],
        "dim_cabeza": info["dim_cabeza"],
        "formas": {k: list(v.shape) for k, v in tensores.items()},
        "dtype": "float32 en disco, float64 al cargar",
        "sha256": sha256(fichero),
        "bytes": fichero.stat().st_size,
    }
    (out / f"sector_vo_{args.modelo}.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False))

    print(f"\n[extraer] {fichero} "
          f"({fichero.stat().st_size / 1e6:.1f} MB)")
    print(f"[extraer] sha256 {manifiesto['sha256'][:16]}...")
    for k, v in tensores.items():
        print(f"[extraer]   {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
