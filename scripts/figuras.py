"""tres figuras desde el csv del barrido, una por pregunta.

fig1 (q1/q2): distribución de err_circuito por régimen, agregada por
capa, con la identidad marcada como referencia.
fig2 (cola gl): err_circuito por escala de gl_e — la cola de escala
baja degradando (o no) es la conexión con el incidente del r
singular del paper 1.
fig3 (q3): mejor-de-k por-cabeza vs compartida, por capa.

uso:
    python scripts/figuras.py --modelo pythia
    python scripts/figuras.py --modelo vitb
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENTRADA = "artifacts/logs/quant_orbita/quant_orbita.csv"
SALIDA_DIR = Path("artifacts/figs")


def fig_regimenes(df: pd.DataFrame, modelo: str, bits: int) -> None:
    """distribución de e por régimen (q1/q2), agregando cabezas y capas.

    la identidad se marca además como línea horizontal de referencia
    sobre los demás regímenes: hace visible q2 (¿el gauge entrenado
    está cerca o lejos de lo muestreado?) de un vistazo, no solo como
    una caja más en la comparación.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    orden = ["identidad", "orto_comp", "orto_ph", "gl_e"]
    datos = [df[(df.regimen == r) & (df.bits == bits)].err_circuito.values
            for r in orden]
    ax.boxplot(datos, tick_labels=orden, showfliers=False)
    mediana_identidad = df[(df.regimen == "identidad")
                           & (df.bits == bits)].err_circuito.median()
    ax.axhline(mediana_identidad, color="crimson", linestyle="--",
              linewidth=1, label="mediana identidad (referencia)")
    ax.set_ylabel("error relativo del circuito $e$")
    ax.set_title(f"{modelo}: $e$ por régimen de gauge (bits={bits})")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA_DIR / f"{modelo}_regimenes_bits{bits}.png", dpi=150)
    plt.close(fig)


def fig_cola_gl(df: pd.DataFrame, modelo: str, bits: int) -> None:
    """percentiles de e por escala de gl_e — ¿degrada la escala baja?

    NO se usa boxplot sin outliers aquí: con mediana 2,85 y máximo
    3,3e5 (pythia, escala 2, bits=4), esconder la cola es esconder el
    titular de la nota. se reportan p50/p90/p99/máx explícitos, en
    escala log, uno por escala de gauge.
    """
    gl = df[(df.regimen == "gl_e") & (df.bits == bits)]
    escalas = sorted(gl.escala.unique())
    percentiles = [0.50, 0.90, 0.99, 1.00]
    etiquetas = ["p50", "p90", "p99", "máx"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = range(len(escalas))
    ancho = 0.2
    for i, (p, etq) in enumerate(zip(percentiles, etiquetas)):
        vals = [gl[gl.escala == e].err_circuito.quantile(p)
               for e in escalas]
        ax.bar([xi + i * ancho for xi in x], vals, width=ancho, label=etq)
    ax.set_xticks([xi + 1.5 * ancho for xi in x])
    ax.set_xticklabels([str(e) for e in escalas])
    ax.set_xlabel("escala de r = randn + escala·i")
    ax.set_ylabel("error relativo del circuito $e$")
    ax.set_title(f"{modelo}: cola gl_e por escala, percentiles "
                f"(bits={bits})")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA_DIR / f"{modelo}_cola_gl_bits{bits}.png", dpi=150)
    plt.close(fig)


def fig_por_cabeza_vs_compartida(df: pd.DataFrame, modelo: str,
                                 bits: int) -> None:
    """mejor-de-k por-cabeza vs compartida, por capa (q3)."""
    sub = df[(df.bits == bits) & df.regimen.isin(["orto_ph", "orto_comp"])]
    mejor = sub.groupby(
        ["capa", "cabeza", "regimen"]).err_circuito.min().reset_index()
    piv = mejor.pivot_table(index=["capa", "cabeza"], columns="regimen",
                            values="err_circuito")
    por_capa = piv.groupby("capa").median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(por_capa.index, por_capa["orto_ph"], marker="o",
           label="por-cabeza (mejor-de-k)")
    ax.plot(por_capa.index, por_capa["orto_comp"], marker="s",
           label="compartida (mejor-de-k)")
    ax.set_xlabel("capa")
    ax.set_ylabel("error relativo del circuito $e$ (mediana)")
    ax.set_title(f"{modelo}: por-cabeza vs compartida (bits={bits})")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA_DIR / f"{modelo}_por_cabeza_vs_compartida_"
               f"bits{bits}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """genera las tres figuras (a bits=4, el régimen de interés)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(ENTRADA)
    df = df[df.modelo == args.modelo]
    if df.empty:
        raise SystemExit(f"sin filas para modelo={args.modelo}")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    fig_regimenes(df, args.modelo, args.bits)
    fig_cola_gl(df, args.modelo, args.bits)
    fig_por_cabeza_vs_compartida(df, args.modelo, args.bits)
    print(f"[figuras] guardadas en {SALIDA_DIR}/{args.modelo}_*.png")


if __name__ == "__main__":
    main()
