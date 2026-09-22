"""certifica la demo celda a celda contra los barridos publicados.

la demo no reimplementa nada: monta el portador de
`demo/portador.py` y deja correr `src/gauges.py`, `src/quantizer.py`
y `src/metrics.py` sin un byte cambiado. eso se comprueba con
números, y admite la forma fuerte: los barridos guardan cada celda
con su semilla, así que la demo puede recorrer exactamente las mismas
y contrastar valor contra valor.

tres compuertas.

* **manifiesto** — sha256 y formas del portador contra su json. si
  esto falla, el portador no es el que se extrajo y lo demás sobra.
* **barrido de regímenes** — celdas de `quant_orbita.csv` con su
  gauge reconstruido por replay del generador. mismos tensores, misma
  aritmética: han de coincidir a precisión de máquina.
* **punto balanceado** — celdas de `punto_balanceado.csv`, que no
  llevan azar: el gauge se construye por svd del circuito. es la
  compuerta más estricta de las tres.

se corre contra el árbol desplegado, no contra el repo: el Space
lleva su propia copia de `src/` y es esa la que atiende visitantes.

uso:
    python scripts/paridad_demo.py
    python scripts/paridad_demo.py --raiz /ruta/al/arbol/desplegado
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.portador import carga, par, verifica_manifiesto
from src.gauges import aplica_gauge, gauge_balanceado, producto_normas
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza
from src.reconstruccion import reconstruye_muchos

CSV_ORBITA = "artifacts/logs/quant_orbita/quant_orbita.csv"
CSV_BALANCE = "artifacts/logs/quant_kappa/punto_balanceado.csv"
MODELOS = {"pythia": {"n_capas": 24, "n_cabezas": 16},
           "vitb": {"n_capas": 12, "n_cabezas": 12}}
DIM_CABEZA = 64
TOL = 1e-12
N_CELDAS = 60


def compuerta_manifiesto(modelos: list[str], raiz: Path | None) -> None:
    """sha256 y formas del portador contra su manifiesto."""
    for m in modelos:
        verifica_manifiesto(m, raiz)
        print(f"[paridad] manifiesto de {m}: ok")


def compuerta_orbita(
    modelo: str,
    raiz: Path | None,
    n: int = N_CELDAS,
    semilla: int = 0,
) -> float:
    """celdas del barrido de regímenes, con el gauge por replay.

    args:
        modelo: 'vitb' o 'pythia'.
        raiz: directorio del portador.
        n: celdas a contrastar.
        semilla: semilla del muestreo de celdas.

    returns:
        la peor diferencia absoluta hallada.

    raises:
        RuntimeError: si alguna supera la tolerancia.
    """
    df = pd.read_csv(CSV_ORBITA)
    df = df[(df.modelo == modelo) & (df.regimen != "identidad")]
    df["escala"] = pd.to_numeric(df["escala"], errors="coerce")
    muestra = df.sample(n, random_state=semilla)
    info = MODELOS[modelo]
    objetivos = set()
    for _, f in muestra.iterrows():
        esc = None if pd.isna(f.escala) else float(f.escala)
        objetivos.add((int(f.capa), int(f.cabeza), f.regimen,
                       int(f.muestra), esc))
    rs = reconstruye_muchos(objetivos, info["n_capas"],
                            info["n_cabezas"], DIM_CABEZA)
    port = carga(modelo, raiz)
    peor = 0.0
    for _, f in tqdm(list(muestra.iterrows()),
                     desc=f"paridad {modelo}"):
        esc = None if pd.isna(f.escala) else float(f.escala)
        r = rs[(int(f.capa), int(f.cabeza), f.regimen,
                int(f.muestra), esc)]
        w_v, w_o = par(port, int(f.capa), int(f.cabeza))
        m_ref = circuito(w_v, w_o)
        a, b = aplica_gauge(w_v, w_o, r)
        e = error_relativo(rtn_cuantiza_descuantiza(a, int(f.bits)),
                           rtn_cuantiza_descuantiza(b, int(f.bits)),
                           m_ref)
        peor = max(peor, abs(e - float(f.err_circuito)))
    if peor >= TOL:
        raise RuntimeError(
            f"paridad del barrido fallida en {modelo}: peor diferencia "
            f"{peor:.2e} >= {TOL:.0e}")
    print(f"[paridad] barrido de regímenes, {modelo}: {n} celdas, "
          f"peor diferencia {peor:.2e}")
    return peor


def compuerta_balanceado(
    modelo: str,
    raiz: Path | None,
    n: int = N_CELDAS,
    semilla: int = 0,
) -> float:
    """celdas del punto balanceado, sin azar de por medio.

    args:
        modelo: 'vitb' o 'pythia'.
        raiz: directorio del portador.
        n: celdas a contrastar.
        semilla: semilla del muestreo de celdas.

    returns:
        la peor diferencia relativa hallada.

    raises:
        RuntimeError: si alguna supera la tolerancia.
    """
    df = pd.read_csv(CSV_BALANCE)
    df = df[df.modelo == modelo]
    muestra = df.sample(min(n, len(df)), random_state=semilla)
    port = carga(modelo, raiz)
    peor = 0.0
    for _, f in tqdm(list(muestra.iterrows()),
                     desc=f"balanceado {modelo}"):
        w_v, w_o = par(port, int(f.capa), int(f.cabeza))
        m_ref = circuito(w_v, w_o)
        r, p_bal = gauge_balanceado(w_v, w_o)
        a, b = aplica_gauge(w_v, w_o, r)
        e_bal = error_relativo(rtn_cuantiza_descuantiza(a, int(f.bits)),
                               rtn_cuantiza_descuantiza(b, int(f.bits)),
                               m_ref)
        peor = max(peor, abs(e_bal - float(f.e_bal)),
                   abs(p_bal - float(f.p_bal)),
                   abs(producto_normas(w_v, w_o, r) - float(f.p_bal)))
    if peor >= TOL:
        raise RuntimeError(
            f"paridad del punto balanceado fallida en {modelo}: peor "
            f"diferencia {peor:.2e} >= {TOL:.0e}")
    print(f"[paridad] punto balanceado, {modelo}: "
          f"{len(muestra)} celdas, peor diferencia {peor:.2e}")
    return peor


def main() -> None:
    """corre las tres compuertas sobre los dos modelos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", default=None,
                        help="directorio del portador, si no el del repo")
    parser.add_argument("--celdas", type=int, default=N_CELDAS)
    args = parser.parse_args()
    raiz = Path(args.raiz) if args.raiz else None
    modelos = list(MODELOS)

    compuerta_manifiesto(modelos, raiz)
    peores = []
    for m in modelos:
        peores.append(compuerta_orbita(m, raiz, args.celdas))
        peores.append(compuerta_balanceado(m, raiz, args.celdas))
    print(f"\n[paridad] TODAS LAS COMPUERTAS PASAN. peor diferencia "
          f"global {max(peores):.2e} (tolerancia {TOL:.0e}).")


if __name__ == "__main__":
    main()
