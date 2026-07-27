"""barrido de cuantización a lo largo de la órbita de gauge.

para cada capa y cabeza de un modelo, cuantiza (w_v, w_o) bajo cuatro
regímenes de gauge —identidad, ortogonal por-cabeza, ortogonal
compartida entre cabezas de la capa, y gl general en una malla de
escalas— y mide el error relativo del circuito ov recompuesto en
fp64. lectura en frío: no se interpreta nada hasta completar el
barrido entero de un modelo (ver scripts/lectura_fria.py).

uso:
    python scripts/sweep.py --modelo pythia --config configs/checkpoints.yaml
    python scripts/sweep.py --modelo vitb --config configs/checkpoints.yaml
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
from src.gauges import gl_general, ortogonal
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza
from src.weights import (cargar_pythia, cargar_vitb, w_v_w_o_pythia,
                         w_v_w_o_vitb)

K_MUESTRAS = 32
ESCALAS_GL = [2.0, 8.0, 32.0]
BITS = [4, 8]
SALIDA = "artifacts/logs/quant_orbita/quant_orbita.csv"
CAMPOS = ["modelo", "capa", "cabeza", "regimen", "escala", "muestra",
         "bits", "err_circuito"]

MODELOS = {
    "pythia": {"n_capas": 24, "n_cabezas": 16, "dim_cabeza": 64},
    "vitb": {"n_capas": 12, "n_cabezas": 12, "dim_cabeza": 64},
}


def _carga(nombre: str, cfg: dict):
    """carga el modelo y su función de extracción por-cabeza."""
    if nombre == "pythia":
        return cargar_pythia(cfg["pythia_model_id"]), w_v_w_o_pythia
    return cargar_vitb(cfg["vitb_ckpt"]), w_v_w_o_vitb


def _mide(
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    r: torch.Tensor,
    m_ref: torch.Tensor,
    bits: int,
) -> float:
    """aplica el gauge r, cuantiza a `bits` y devuelve el error."""
    r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
    w_v_g = r.transpose(-2, -1) @ w_v_h
    w_o_g = w_o_h @ r_inv_t
    w_v_q = rtn_cuantiza_descuantiza(w_v_g, bits)
    w_o_q = rtn_cuantiza_descuantiza(w_o_g, bits)
    return error_relativo(w_v_q, w_o_q, m_ref)


def barre_capa(
    nombre_modelo: str,
    capa: int,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    dim_cabeza: int,
    gen: torch.Generator,
) -> list[dict]:
    """barre los cuatro regímenes sobre todas las cabezas de una capa.

    args:
        nombre_modelo: 'pythia' o 'vitb'.
        capa: índice de capa.
        w_v: [h, d_h, d] en float64.
        w_o: [h, d, d_h] en float64.
        dim_cabeza: d_h.
        gen: generador de torch, avanza con cada muestra (no se
            reinicia por cabeza: las muestras de una capa son una
            secuencia única, reproducible por semilla global).

    returns:
        lista de filas (dicts) con todas las mediciones de la capa.
    """
    n_cabezas = w_v.shape[0]
    filas: list[dict] = []
    identidad = torch.eye(dim_cabeza, dtype=torch.float64)

    # gauges ortogonales compartidos entre cabezas: se muestrean una
    # vez por capa, antes del bucle de cabezas, para que las k=32
    # muestras sean las MISMAS r en cada cabeza (régimen orto_comp).
    r_compartidos = [ortogonal(dim_cabeza, gen) for _ in range(K_MUESTRAS)]

    for h in range(n_cabezas):
        m_ref = circuito(w_v[h], w_o[h])
        for bits in BITS:
            filas.append({
                "modelo": nombre_modelo, "capa": capa, "cabeza": h,
                "regimen": "identidad", "escala": "", "muestra": 0,
                "bits": bits,
                "err_circuito": _mide(w_v[h], w_o[h], identidad,
                                      m_ref, bits),
            })
        for m in range(K_MUESTRAS):
            r = ortogonal(dim_cabeza, gen)
            for bits in BITS:
                filas.append({
                    "modelo": nombre_modelo, "capa": capa, "cabeza": h,
                    "regimen": "orto_ph", "escala": "", "muestra": m,
                    "bits": bits,
                    "err_circuito": _mide(w_v[h], w_o[h], r, m_ref, bits),
                })
        for m, r in enumerate(r_compartidos):
            for bits in BITS:
                filas.append({
                    "modelo": nombre_modelo, "capa": capa, "cabeza": h,
                    "regimen": "orto_comp", "escala": "", "muestra": m,
                    "bits": bits,
                    "err_circuito": _mide(w_v[h], w_o[h], r, m_ref, bits),
                })
        for escala in ESCALAS_GL:
            for m in range(K_MUESTRAS):
                r = gl_general(dim_cabeza, escala, gen)
                for bits in BITS:
                    filas.append({
                        "modelo": nombre_modelo, "capa": capa,
                        "cabeza": h, "regimen": "gl_e", "escala": escala,
                        "muestra": m, "bits": bits,
                        "err_circuito": _mide(w_v[h], w_o[h], r, m_ref,
                                              bits),
                    })
    return filas


def main() -> None:
    """corre el barrido completo de un modelo y lo vuelca al csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True,
                        choices=list(MODELOS.keys()))
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = MODELOS[args.modelo]

    print(f"[sweep] cargando {args.modelo}...", flush=True)
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
    print(f"[sweep] {args.modelo}: {len(filas)} filas -> {out}")

    contador = Counter(
        (f["capa"], f["cabeza"], f["regimen"], f["escala"], f["muestra"],
         f["bits"])
        for f in filas)
    esperado = (info["n_capas"] * info["n_cabezas"]
               * (1 + K_MUESTRAS + K_MUESTRAS  # identidad+orto_ph+orto_comp
                  + K_MUESTRAS * len(ESCALAS_GL))  # gl_e
               * len(BITS))
    duplicados = sum(v - 1 for v in contador.values() if v > 1)
    print(f"[sweep] claves únicas: {len(contador)} "
         f"(filas {sum(contador.values())}, esperado {esperado}, "
         f"duplicados {duplicados})")

    rng = random.Random(0)
    print("\n[sweep] 15 observaciones aleatorias:")
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)


if __name__ == "__main__":
    main()
