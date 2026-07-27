"""compuerta del barrido: tres sanities, sin las cuales no hay dato.

1. gauge ortogonal SIN cuantizar -> e en el suelo fp64 (~1e-13); si
   no, la extracción de (w_v, w_o) o el gauge están rotos.
2. int8 sobre identidad -> e un orden de magnitud menor que int4.
3. cuantizar-descuantizar-recuantizar es idempotente.

se corren sobre una muestra pequeña de capas/cabezas de los dos
modelos (pythia-410m, vit-b seed42): la compuerta es sobre el
pipeline, no sobre el barrido completo.

uso:
    python scripts/sanity.py --config configs/checkpoints.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.gauges import ortogonal
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza
from src.weights import (cargar_pythia, cargar_vitb, w_v_w_o_pythia,
                         w_v_w_o_vitb)

CAPAS_MUESTRA = [0, 1]
CABEZAS_MUESTRA = [0, 1, 2]
SUELO_FP64 = 1e-10  # margen holgado sobre el ~1e-13 esperado


def _modelos_muestra(cfg: dict) -> list[tuple[str, object, int, int]]:
    """carga los dos modelos y devuelve (nombre, modelo, h, dh)."""
    print("[sanity] cargando pythia-410m...", flush=True)
    pythia = cargar_pythia(cfg["pythia_model_id"])
    print("[sanity] cargando vit-b seed42...", flush=True)
    vitb = cargar_vitb(cfg["vitb_ckpt"])
    return [
        ("pythia", pythia, 16, 64),
        ("vitb", vitb, 12, 64),
    ]


def _extrae(nombre: str, modelo, capa: int):
    """despacha a la extracción por-cabeza del modelo correspondiente."""
    if nombre == "pythia":
        return w_v_w_o_pythia(modelo, capa)
    return w_v_w_o_vitb(modelo, capa)


def sanity_gauge_sin_cuantizar(
    modelos: list[tuple[str, object, int, int]],
) -> bool:
    """gauge ortogonal sin cuantizar debe dar e en el suelo fp64."""
    gen = torch.Generator().manual_seed(0)
    peor = 0.0
    for nombre, modelo, _, dim_cabeza in modelos:
        for capa in CAPAS_MUESTRA:
            w_v, w_o = _extrae(nombre, modelo, capa)
            for h in CABEZAS_MUESTRA:
                m_ref = circuito(w_v[h], w_o[h])
                r = ortogonal(dim_cabeza, gen)
                r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
                w_v_g = r.transpose(-2, -1) @ w_v[h]
                w_o_g = w_o[h] @ r_inv_t
                e = error_relativo(w_v_g, w_o_g, m_ref)
                peor = max(peor, e)
    ok = peor < SUELO_FP64
    print(f"[1/3] gauge sin cuantizar: peor e = {peor:.3e} "
          f"(umbral {SUELO_FP64:.0e}) -> {'OK' if ok else 'FALLO'}")
    return ok


def sanity_int8_bate_int4(
    modelos: list[tuple[str, object, int, int]],
) -> bool:
    """int8 sobre identidad debe dar e un orden de magnitud < int4."""
    ok_total = True
    for nombre, modelo, _, _ in modelos:
        w_v, w_o = _extrae(nombre, modelo, CAPAS_MUESTRA[0])
        es_int4, es_int8 = [], []
        for h in CABEZAS_MUESTRA:
            m_ref = circuito(w_v[h], w_o[h])
            w_v4 = rtn_cuantiza_descuantiza(w_v[h], 4)
            w_o4 = rtn_cuantiza_descuantiza(w_o[h], 4)
            w_v8 = rtn_cuantiza_descuantiza(w_v[h], 8)
            w_o8 = rtn_cuantiza_descuantiza(w_o[h], 8)
            es_int4.append(error_relativo(w_v4, w_o4, m_ref))
            es_int8.append(error_relativo(w_v8, w_o8, m_ref))
        m4 = sum(es_int4) / len(es_int4)
        m8 = sum(es_int8) / len(es_int8)
        ok = m8 < m4 / 5  # "un orden de magnitud" con margen
        ok_total &= ok
        print(f"[2/3] {nombre}: e_int4={m4:.3e}  e_int8={m8:.3e} "
              f"(ratio {m4 / m8:.1f}x) -> {'OK' if ok else 'FALLO'}")
    return ok_total


def sanity_idempotencia(
    modelos: list[tuple[str, object, int, int]],
) -> bool:
    """cuantizar-descuantizar-recuantizar debe ser idempotente."""
    ok_total = True
    for nombre, modelo, _, _ in modelos:
        w_v, w_o = _extrae(nombre, modelo, CAPAS_MUESTRA[0])
        for bits in (4, 8):
            w1 = rtn_cuantiza_descuantiza(w_v[0], bits)
            w2 = rtn_cuantiza_descuantiza(w1, bits)
            dif = float((w1 - w2).abs().max())
            ok = dif < 1e-9
            ok_total &= ok
            print(f"[3/3] {nombre} bits={bits}: "
                  f"|requant - quant| max = {dif:.3e} "
                  f"-> {'OK' if ok else 'FALLO'}")
    return ok_total


def main() -> None:
    """corre las tres compuertas y sale con error si alguna falla."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    modelos = _modelos_muestra(cfg)
    resultados = [
        sanity_gauge_sin_cuantizar(modelos),
        sanity_int8_bate_int4(modelos),
        sanity_idempotencia(modelos),
    ]
    if all(resultados):
        print("\n[sanity] las tres compuertas pasan. barrido habilitado.")
        sys.exit(0)
    print("\n[sanity] AL MENOS UNA COMPUERTA FALLA. no correr el barrido.")
    sys.exit(1)


if __name__ == "__main__":
    main()
