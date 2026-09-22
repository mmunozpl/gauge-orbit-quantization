"""el punto balanceado como óptimo predicho de la órbita.

el lema del punto balanceado del paper 1 fija, a circuito constante,
la factorización que minimiza ||w_v r||_f^2 + ||r^-1 w_o||_f^2. la ley
del producto de esta nota dice que el error de cuantización escala con
p(r) = ||w_v r||_f ||r^-1 w_o||_f / (||w_v||_f ||w_o||_f). las dos
piezas se encuentran en una identidad: para toda factorización
m = ab se cumple ||m||_* <= ||a||_f ||b||_f, con igualdad exactamente
en el punto balanceado. el punto balanceado es entonces el mínimo del
producto de normas sobre la órbita entera, y la ley predice allí el
menor error alcanzable.

el punto balanceado es un gauge NO ortogonal. si su error baja como
predice p, la frase «la restricción ortogonal es necesaria» queda
refutada por un contraejemplo construido, no muestreado: existe un
gauge fuera del subgrupo ortogonal que cuantiza mejor que el gauge
entrenado. la frontera correcta no es ortogonal / no ortogonal, sino
la que separa la clase conforme ortogonal {c·q} —donde p = 1 exacto,
neutra— de la anisotropía, que puede inflar o desinflar p.

lecturas fijadas ANTES de correr (22-09-2026):

- l1. el punto balanceado mejora al gauge entrenado si e_bal < e_ident
  en >= 90 % de las celdas; «no mejora» si < 50 %.
- l2. la ley del producto extrapola a p < 1 si la mediana de
  (e_bal/e_ident) / p_bal cae en [0,90, 1,10]. el ajuste publicado se
  hizo sobre familias que INFLAN p; esto es extrapolación fuera de
  muestra al lado que las desinfla, y por eso se declara como prueba,
  no como confirmación.
- l3. la neutralidad de la clase conforme ortogonal se comprueba por
  identidad: p(c·q) = 1 a precisión de máquina para toda c y toda q.
- l4. la anisotropía del punto balanceado se reporta por d(r), la
  desviación típica de los valores singulares partida por su norma
  (la distancia a {c·q} de la ecuación (2) del paper 1): d = 0 denota
  conforme ortogonal.

cualquier desenlace se reporta; la nota no tiene rama fallida.

uso:
    python scripts/punto_balanceado.py --modelo pythia
    python scripts/punto_balanceado.py --modelo vitb
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.gauges import (aplica_gauge, gauge_balanceado, ortogonal,
                        producto_normas)
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza
from src.weights import (cargar_pythia, cargar_vitb, w_v_w_o_pythia,
                         w_v_w_o_vitb)

BITS = [4, 8]
SALIDA = "artifacts/logs/quant_kappa/punto_balanceado.csv"
SALIDA_LECTURA = "artifacts/tables/punto_balanceado.csv"
CAMPOS = ["modelo", "capa", "cabeza", "bits", "e_ident", "e_bal",
          "p_bal", "d_r_bal", "cond_r_bal"]
TOL_CONFORME = 1e-12  # p(c·q) = 1 a precisión de máquina (l3)

MODELOS = {
    "pythia": {"n_capas": 24, "n_cabezas": 16, "dim_cabeza": 64},
    "vitb": {"n_capas": 12, "n_cabezas": 12, "dim_cabeza": 64},
}


def _carga(nombre: str, cfg: dict):
    """carga el modelo y su función de extracción por-cabeza."""
    if nombre == "pythia":
        return cargar_pythia(cfg["pythia_model_id"]), w_v_w_o_pythia
    return cargar_vitb(cfg["vitb_ckpt"]), w_v_w_o_vitb


def distancia_conforme(r: torch.Tensor) -> float:
    """d(r): distancia de r a la clase conforme ortogonal {c·q}.

    ecuación (2) del paper 1: desviación típica de los valores
    singulares partida por su norma. vale 0 exactamente si y solo si
    todos los valores singulares coinciden, que es r = c·q.

    args:
        r: tensor [d_h, d_h] en float64.

    returns:
        d(r) >= 0, escalar float.
    """
    s = torch.linalg.svdvals(r)
    return float((s - s.mean()).norm() / s.norm())


def compuerta_conforme(dim_cabeza: int, gen: torch.Generator) -> None:
    """l3: p(c·q) = 1 a precisión de máquina, y d(c·q) = 0.

    es la comprobación que convierte la frontera del artículo en una
    identidad y no en una impresión: la clase inocua del paper 1 es
    exactamente la clase neutra de la ley del producto.

    args:
        dim_cabeza: d_h.
        gen: generador de torch.

    raises:
        RuntimeError: si algún p(c·q) se desvía de 1 por encima de la
            tolerancia.
    """
    w_v = torch.randn(dim_cabeza, 768, generator=gen, dtype=torch.float64)
    w_o = torch.randn(768, dim_cabeza, generator=gen, dtype=torch.float64)
    peor_p, peor_d = 0.0, 0.0
    for _ in range(20):
        q = ortogonal(dim_cabeza, gen)
        for c in (0.01, 0.5, 1.0, 3.0, 100.0):
            r = c * q
            peor_p = max(peor_p, abs(producto_normas(w_v, w_o, r) - 1.0))
            peor_d = max(peor_d, distancia_conforme(r))
    if peor_p >= TOL_CONFORME:
        raise RuntimeError(
            f"compuerta conforme fallida: peor |p-1| = {peor_p:.2e} "
            f">= {TOL_CONFORME:.0e}; la clase conforme ortogonal no es "
            f"neutra para la ley del producto")
    print(f"[balanceado] compuerta conforme ok: peor |p(cQ)-1| = "
          f"{peor_p:.2e}, peor d(cQ) = {peor_d:.2e}")


def barre_capa(
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    modelo_nombre: str,
    capa: int,
    n_cabezas: int,
) -> list[dict]:
    """mide identidad y punto balanceado en todas las cabezas de una capa.

    args:
        w_v: tensor [h, d_h, d] en float64.
        w_o: tensor [h, d, d_h] en float64.
        modelo_nombre: etiqueta del modelo para el csv.
        capa: índice de capa.
        n_cabezas: cabezas h.

    returns:
        lista de filas del csv, una por (cabeza, bits).
    """
    filas = []
    for h in range(n_cabezas):
        w_v_h, w_o_h = w_v[h], w_o[h]
        m_ref = circuito(w_v_h, w_o_h)
        r, p_bal = gauge_balanceado(w_v_h, w_o_h)
        w_v_b, w_o_b = aplica_gauge(w_v_h, w_o_h, r)
        # el gauge ha de dejar el circuito donde estaba: si no, lo que
        # se mide después no es un cambio de coordenadas.
        deriva = float((circuito(w_v_b, w_o_b) - m_ref).norm()
                       / m_ref.norm())
        if deriva > 1e-10:
            raise RuntimeError(
                f"el gauge balanceado movió el circuito en capa={capa} "
                f"cabeza={h}: {deriva:.2e}")
        d_r = distancia_conforme(r)
        cond_r = float(torch.linalg.cond(r))
        for bits in BITS:
            e_ident = error_relativo(
                rtn_cuantiza_descuantiza(w_v_h, bits),
                rtn_cuantiza_descuantiza(w_o_h, bits), m_ref)
            e_bal = error_relativo(
                rtn_cuantiza_descuantiza(w_v_b, bits),
                rtn_cuantiza_descuantiza(w_o_b, bits), m_ref)
            filas.append({
                "modelo": modelo_nombre, "capa": capa, "cabeza": h,
                "bits": bits, "e_ident": e_ident, "e_bal": e_bal,
                "p_bal": p_bal, "d_r_bal": d_r, "cond_r_bal": cond_r})
    return filas


def lectura(df: pd.DataFrame) -> pd.DataFrame:
    """las lecturas l1, l2 y l4 sobre el barrido completo.

    args:
        df: filas del barrido de un modelo.

    returns:
        tabla de lectura, una fila por (modelo, bits).
    """
    df = df.copy()
    df["ratio_medido"] = df.e_bal / df.e_ident
    df["ratio_sobre_ley"] = df.ratio_medido / df.p_bal
    out = []
    for (modelo, bits), g in df.groupby(["modelo", "bits"]):
        frac_mejora = float((g.e_bal < g.e_ident).mean())
        mediana_ley = float(g.ratio_sobre_ley.median())
        out.append({
            "modelo": modelo, "bits": bits, "celdas": len(g),
            "p_bal_mediano": float(g.p_bal.median()),
            "ratio_medido_mediano": float(g.ratio_medido.median()),
            "ratio_sobre_ley_mediano": mediana_ley,
            "frac_e_bal_menor_que_ident": frac_mejora,
            "d_r_bal_mediano": float(g.d_r_bal.median()),
            "cond_r_bal_mediano": float(g.cond_r_bal.median()),
            "l1_veredicto": ("mejora" if frac_mejora >= 0.90
                             else ("no_mejora" if frac_mejora < 0.50
                                   else "entremedias")),
            "l2_veredicto": ("la_ley_extrapola"
                             if 0.90 <= mediana_ley <= 1.10
                             else "la_ley_no_extrapola"),
        })
    return pd.DataFrame(out)


def main() -> None:
    """corre el barrido del punto balanceado y su lectura en frío."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, choices=list(MODELOS))
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = MODELOS[args.modelo]

    gen = torch.Generator().manual_seed(0)
    compuerta_conforme(info["dim_cabeza"], gen)

    print(f"[balanceado] cargando {args.modelo}...", flush=True)
    modelo, extrae = _carga(args.modelo, cfg)

    filas = []
    for capa in tqdm(range(info["n_capas"]), desc=f"{args.modelo}"):
        w_v, w_o = extrae(modelo, capa, info["n_cabezas"],
                          info["dim_cabeza"])
        filas.extend(barre_capa(w_v, w_o, args.modelo, capa,
                                info["n_cabezas"]))

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    previas = []
    if out.exists():
        previas = [f for f in csv.DictReader(out.open())
                   if f["modelo"] != args.modelo]
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CAMPOS)
        wr.writeheader()
        wr.writerows(previas)
        wr.writerows(filas)
    print(f"\n[balanceado] {args.modelo}: {len(filas)} filas -> {out}")

    esperado = info["n_capas"] * info["n_cabezas"] * len(BITS)
    claves = {(f["capa"], f["cabeza"], f["bits"]) for f in filas}
    print(f"[balanceado] claves únicas: {len(claves)} "
          f"(esperado {esperado})")
    if len(claves) != esperado or len(filas) != esperado:
        raise RuntimeError("el barrido tiene huecos o duplicados")

    print("\n[balanceado] 15 observaciones aleatorias:")
    for f in random.sample(filas, min(15, len(filas))):
        print(f"  {f}")

    df = pd.DataFrame(filas)
    tabla = lectura(df)
    out_l = Path(SALIDA_LECTURA)
    out_l.parent.mkdir(parents=True, exist_ok=True)
    if out_l.exists():
        antes = pd.read_csv(out_l)
        tabla = pd.concat(
            [antes[antes.modelo != args.modelo], tabla], ignore_index=True)
    tabla.to_csv(out_l, index=False)

    print(f"\n=== lectura en frío — punto balanceado, {args.modelo} ===")
    print(tabla[tabla.modelo == args.modelo].to_string(index=False))
    print(f"\n[guardado] lectura en {out_l}")


if __name__ == "__main__":
    main()
