"""barrido fino del error de cuantización contra el número de condición.

la nota afirma que el mecanismo de la cola tóxica es de
condicionamiento, y hoy lo sostiene sobre tres escalas cuyos rangos de
kappa se solapan. este barrido controla kappa directamente: para cada
capa y cabeza, aplica gauges con condicionamiento fijado sobre una
rejilla log-espaciada, cuantiza y mide el error del circuito ov
recompuesto en fp64.

lectura en frío: no se interpreta nada hasta completar el barrido
entero de un modelo (ver scripts/lectura_kappa.py), y contra la regla
preregistrada por escrito.

uso:
    python scripts/sweep_kappa.py --modelo pythia
    python scripts/sweep_kappa.py --modelo vitb
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
from src.gauges import gl_condicionado
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza
from src.weights import (cargar_pythia, cargar_vitb, w_v_w_o_pythia,
                         w_v_w_o_vitb)

K_MUESTRAS = 32
BITS = [4, 8]
TOL_KAPPA = 1e-10  # compuerta: kappa medido contra objetivo
SALIDA = "artifacts/logs/quant_kappa/quant_kappa.csv"
CAMPOS = ["modelo", "capa", "cabeza", "kappa", "muestra", "bits",
          "err_circuito", "norma_wv", "norma_wo"]

MODELOS = {
    "pythia": {"n_capas": 24, "n_cabezas": 16, "dim_cabeza": 64},
    "vitb": {"n_capas": 12, "n_cabezas": 12, "dim_cabeza": 64},
}


def rejilla_kappa() -> list[float]:
    """rejilla preregistrada: densa bajo 200, con cola hasta 5000.

    quince puntos geométricos entre 1 y 200 —la zona donde el ajuste
    tiene señal, según el rango de e que la enmienda e2 fija— y cuatro
    por encima para documentar la cola.

    returns:
        lista de kappas objetivo, ascendente.
    """
    densa = torch.logspace(0.0, torch.log10(torch.tensor(200.0)), 15,
                           dtype=torch.float64)
    cola = [400.0, 1000.0, 2500.0, 5000.0]
    return [round(float(k), 6) for k in densa] + cola


def _mide_ambos_bits(
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    r: torch.Tensor,
    m_ref: torch.Tensor,
) -> dict:
    """aplica el gauge una vez y cuantiza a las dos anchuras.

    se registran además las normas de frobenius de los dos factores ya
    transformados: son la medición secundaria descriptiva del
    preregistro, que comprueba el mecanismo —el condicionamiento
    inflando ambos factores a la vez— de forma directa.

    args:
        w_v_h: tensor [d_h, d] en float64.
        w_o_h: tensor [d, d_h] en float64.
        r: gauge [d_h, d_h] en float64.
        m_ref: circuito exacto de referencia, [d, d] en float64.

    returns:
        dict con el error por anchura y las dos normas.
    """
    r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
    w_v_g = r.transpose(-2, -1) @ w_v_h
    w_o_g = w_o_h @ r_inv_t
    salida = {"norma_wv": float(w_v_g.norm()),
              "norma_wo": float(w_o_g.norm())}
    for bits in BITS:
        salida[bits] = error_relativo(
            rtn_cuantiza_descuantiza(w_v_g, bits),
            rtn_cuantiza_descuantiza(w_o_g, bits),
            m_ref)
    return salida


def barre_capa(
    nombre_modelo: str,
    capa: int,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    dim_cabeza: int,
    kappas: list[float],
    gen: torch.Generator,
) -> list[dict]:
    """barre la rejilla de kappa sobre todas las cabezas de una capa.

    args:
        nombre_modelo: 'pythia' o 'vitb'.
        capa: índice de capa.
        w_v: [h, d_h, d] en float64.
        w_o: [h, d, d_h] en float64.
        dim_cabeza: d_h.
        kappas: rejilla de números de condición objetivo.
        gen: generador de torch; avanza con cada muestra, sin
            reiniciarse por cabeza.

    returns:
        lista de filas con todas las mediciones de la capa.
    """
    filas: list[dict] = []
    for h in range(w_v.shape[0]):
        m_ref = circuito(w_v[h], w_o[h])
        for kappa in kappas:
            for m in range(K_MUESTRAS):
                r = gl_condicionado(dim_cabeza, kappa, gen)
                med = _mide_ambos_bits(w_v[h], w_o[h], r, m_ref)
                for bits in BITS:
                    filas.append({
                        "modelo": nombre_modelo, "capa": capa,
                        "cabeza": h, "kappa": kappa, "muestra": m,
                        "bits": bits, "err_circuito": med[bits],
                        "norma_wv": med["norma_wv"],
                        "norma_wo": med["norma_wo"],
                    })
    return filas


def compuerta_kappa(dim_cabeza: int, kappas: list[float],
                    gen: torch.Generator) -> None:
    """verifica que cond(r) reproduce el objetivo; aborta si no.

    args:
        dim_cabeza: d_h.
        kappas: rejilla a verificar.
        gen: generador de torch.

    raises:
        RuntimeError: si algún kappa medido se desvía del objetivo por
            encima de la tolerancia.
    """
    peor = 0.0
    for kappa in kappas:
        for _ in range(4):
            medido = float(torch.linalg.cond(
                gl_condicionado(dim_cabeza, kappa, gen)))
            peor = max(peor, abs(medido - kappa) / kappa)
    if peor >= TOL_KAPPA:
        raise RuntimeError(
            f"compuerta de kappa fallida: peor error relativo {peor:.2e} "
            f">= {TOL_KAPPA:.0e}; el barrido no corre")
    print(f"[sweep_kappa] compuerta ok: peor error relativo "
          f"{peor:.2e} < {TOL_KAPPA:.0e}")


def _carga(nombre: str, cfg: dict):
    """carga el modelo y su función de extracción por-cabeza."""
    if nombre == "pythia":
        return cargar_pythia(cfg["pythia_model_id"]), w_v_w_o_pythia
    return cargar_vitb(cfg["vitb_ckpt"]), w_v_w_o_vitb


def main() -> None:
    """corre el barrido de kappa de un modelo y lo vuelca al csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True,
                        choices=list(MODELOS.keys()))
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--capas", type=int, default=None,
                        help="limita el número de capas (solo para "
                             "estimar coste, no para el barrido real)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = MODELOS[args.modelo]
    kappas = rejilla_kappa()

    gen = torch.Generator().manual_seed(args.semilla)
    compuerta_kappa(info["dim_cabeza"], kappas, gen)

    print(f"[sweep_kappa] cargando {args.modelo}...", flush=True)
    modelo, extrae = _carga(args.modelo, cfg)
    gen = torch.Generator().manual_seed(args.semilla)

    n_capas = args.capas or info["n_capas"]
    filas: list[dict] = []
    for capa in tqdm(range(n_capas), desc=f"{args.modelo} capas"):
        w_v, w_o = extrae(modelo, capa, info["n_cabezas"],
                          info["dim_cabeza"])
        filas.extend(barre_capa(args.modelo, capa, w_v, w_o,
                                info["dim_cabeza"], kappas, gen))

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    existe = out.exists()
    with out.open("a" if existe else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CAMPOS)
        if not existe:
            wr.writeheader()
        wr.writerows(filas)
    print(f"[sweep_kappa] {args.modelo}: {len(filas)} filas -> {out}")

    contador = Counter(
        (f["capa"], f["cabeza"], f["kappa"], f["muestra"], f["bits"])
        for f in filas)
    esperado = (n_capas * info["n_cabezas"] * len(kappas) * K_MUESTRAS
                * len(BITS))
    duplicados = sum(v - 1 for v in contador.values() if v > 1)
    print(f"[sweep_kappa] claves únicas: {len(contador)} "
          f"(filas {sum(contador.values())}, esperado {esperado}, "
          f"duplicados {duplicados})")

    rng = random.Random(0)
    print("\n[sweep_kappa] 15 observaciones aleatorias:")
    for fila in rng.sample(filas, min(15, len(filas))):
        print(fila)


if __name__ == "__main__":
    main()
