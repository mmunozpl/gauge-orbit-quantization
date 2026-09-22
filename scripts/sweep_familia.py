"""re-barrido de la familia original de gauges, registrando normas.

el barrido original (scripts/sweep.py) no guardó las normas de los
factores transformados, y sin ellas no se puede contrastar el colapso
entre familias: dentro de la familia controlada el producto de normas
es función determinista de kappa, así que el colapso solo se afirma
comparando familias distintas al mismo producto.

se re-corre la malla de escalas {2, 8, 32} con `_mide_ambos_bits`, más
una fila de identidad por celda que sirve de referencia común a las dos
familias (e_0 y el producto de normas sin gauge).

uso:
    python scripts/sweep_familia.py --modelo vitb
    python scripts/sweep_familia.py --modelo pythia
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.gauges import gl_general
from src.metrics import circuito
from sweep_kappa import (BITS, K_MUESTRAS, MODELOS, _carga,
                         _mide_ambos_bits)

ESCALAS_GL = [2.0, 8.0, 32.0]
SALIDA = "artifacts/logs/quant_kappa/quant_familia.csv"
CAMPOS = ["modelo", "capa", "cabeza", "regimen", "escala", "muestra",
          "bits", "err_circuito", "kappa", "norma_wv", "norma_wo"]


def barre_capa(
    nombre_modelo: str,
    capa: int,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    dim_cabeza: int,
    gen: torch.Generator,
) -> list[dict]:
    """barre identidad y la malla de escalas sobre las cabezas de una capa.

    args:
        nombre_modelo: 'pythia' o 'vitb'.
        capa: índice de capa.
        w_v: [h, d_h, d] en float64.
        w_o: [h, d, d_h] en float64.
        dim_cabeza: d_h.
        gen: generador de torch, avanza con cada muestra.

    returns:
        lista de filas con error, kappa y las dos normas.
    """
    filas: list[dict] = []
    identidad = torch.eye(dim_cabeza, dtype=torch.float64)
    for h in range(w_v.shape[0]):
        m_ref = circuito(w_v[h], w_o[h])
        base = _mide_ambos_bits(w_v[h], w_o[h], identidad, m_ref)
        for bits in BITS:
            filas.append({
                "modelo": nombre_modelo, "capa": capa, "cabeza": h,
                "regimen": "identidad", "escala": "", "muestra": 0,
                "bits": bits, "err_circuito": base[bits], "kappa": 1.0,
                "norma_wv": base["norma_wv"],
                "norma_wo": base["norma_wo"]})
        for escala in ESCALAS_GL:
            for m in range(K_MUESTRAS):
                r = gl_general(dim_cabeza, escala, gen)
                kappa = float(torch.linalg.cond(r))
                med = _mide_ambos_bits(w_v[h], w_o[h], r, m_ref)
                for bits in BITS:
                    filas.append({
                        "modelo": nombre_modelo, "capa": capa,
                        "cabeza": h, "regimen": "gl_e", "escala": escala,
                        "muestra": m, "bits": bits,
                        "err_circuito": med[bits], "kappa": kappa,
                        "norma_wv": med["norma_wv"],
                        "norma_wo": med["norma_wo"]})
    return filas


def main() -> None:
    """corre el re-barrido de un modelo y lo vuelca al csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True,
                        choices=list(MODELOS.keys()))
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = MODELOS[args.modelo]

    print(f"[sweep_familia] cargando {args.modelo}...", flush=True)
    modelo, extrae = _carga(args.modelo, cfg)
    gen = torch.Generator().manual_seed(args.semilla)

    filas: list[dict] = []
    for capa in tqdm(range(info["n_capas"]), desc=f"{args.modelo} capas"):
        w_v, w_o = extrae(modelo, capa, info["n_cabezas"],
                          info["dim_cabeza"])
        filas.extend(barre_capa(args.modelo, capa, w_v, w_o,
                                info["dim_cabeza"], gen))

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    existe = out.exists()
    with out.open("a" if existe else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CAMPOS)
        if not existe:
            wr.writeheader()
        wr.writerows(filas)
    print(f"[sweep_familia] {args.modelo}: {len(filas)} filas -> {out}")

    contador = Counter(
        (f["capa"], f["cabeza"], f["regimen"], f["escala"], f["muestra"],
         f["bits"]) for f in filas)
    esperado = (info["n_capas"] * info["n_cabezas"]
                * (1 + K_MUESTRAS * len(ESCALAS_GL)) * len(BITS))
    duplicados = sum(v - 1 for v in contador.values() if v > 1)
    print(f"[sweep_familia] claves únicas: {len(contador)} "
          f"(filas {sum(contador.values())}, esperado {esperado}, "
          f"duplicados {duplicados})")

    rng = random.Random(0)
    print("\n[sweep_familia] 15 observaciones aleatorias:")
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)


if __name__ == "__main__":
    main()
