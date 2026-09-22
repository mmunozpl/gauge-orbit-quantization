"""dos figuras del barrido de kappa y del colapso entre familias.

fig1 (por modelo): percentiles explícitos de e contra kappa. misma
convención que la figura de la cola gl —p50/p90/p99/máx, nunca caja
con outliers ocultos—, porque también aquí el titular vive en la cola.

fig2 (una, ambos modelos): el colapso. e/e_0 contra el producto de
normas, las dos familias superpuestas, con la recta de pendiente 1 que
la derivación predice sin ajustar nada. al lado, la misma nube contra
kappa, donde las familias NO colapsan: el contraste es el resultado.

uso:
    python scripts/figuras_kappa.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colapso_producto import CLAVE, referencia

KAPPA_CSV = "artifacts/logs/quant_kappa/quant_kappa.csv"
FAMILIA_CSV = "artifacts/logs/quant_kappa/quant_familia.csv"
SALIDA_DIR = Path("artifacts/figs")
PCT = [50, 90, 99]


def fig_percentiles(df: pd.DataFrame, modelo: str, bits: int) -> None:
    """percentiles de e contra kappa, con el suelo marcado.

    args:
        df: barrido de kappa.
        modelo: nombre del modelo.
        bits: anchura.
    """
    sub = df[(df.modelo == modelo) & (df.bits == bits)]
    g = sub.groupby("kappa").err_circuito
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PCT:
        ax.plot(g.quantile(p / 100).index, g.quantile(p / 100).values,
                marker="o", markersize=3, label=f"p{p}")
    ax.plot(g.max().index, g.max().values, marker="^", markersize=3,
            linestyle="--", label="máx")
    e_0 = float(g.median().loc[g.median().index.min()])
    ax.axhline(e_0, color="crimson", linestyle=":", linewidth=1,
               label=f"suelo $e_0$ = {e_0:.4f}")
    ax.axhline(1.0, color="black", linestyle="-.", linewidth=1,
               label="circuito destruido ($e=1$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"número de condición $\kappa(R)$")
    ax.set_ylabel("error relativo del circuito $e$")
    ax.set_title(f"{modelo}: $e$ contra $\\kappa$ (bits={bits})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA_DIR / f"{modelo}_kappa_bits{bits}.png", dpi=150)
    plt.close(fig)


def fig_colapso(todo: pd.DataFrame, bits: int) -> None:
    """el colapso: e/e_0 contra el producto, y contra kappa al lado.

    args:
        todo: filas de ambas familias con producto y razon_e.
        bits: anchura.
    """
    sub = todo[todo.bits == bits]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    for ax, var, titulo in (
            (axes[0], "kappa", r"contra $\kappa$: no colapsan"),
            (axes[1], "producto", "contra el producto de normas: colapsan")):
        for fam, color in (("original", "tab:blue"),
                           ("controlada", "tab:orange")):
            g = sub[sub.familia == fam]
            ax.scatter(g[var], g.razon_e, s=2, alpha=0.15, color=color,
                       label=f"familia {fam}", rasterized=True)
        x = np.logspace(0, np.log10(sub[var].max()), 50)
        if var == "producto":
            ax.plot(x, x, color="black", linestyle="--", linewidth=1.2,
                    label="pendiente 1 (predicha)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa(R)$" if var == "kappa"
                      else r"$\|W_vR\|\,\|R^{-1}W_O\|\,/\,\|W_v\|\,\|W_O\|$")
        ax.set_title(titulo, fontsize=10)
        leyenda = ax.legend(fontsize=8, markerscale=4)
        for h in leyenda.legend_handles:
            h.set_alpha(1.0)
    axes[0].set_ylabel(r"$e/e_0$")
    fig.suptitle(f"colapso entre familias de gauge (bits={bits})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDA_DIR / f"colapso_producto_bits{bits}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """genera las figuras del barrido de kappa y del colapso."""
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    kap = pd.read_csv(KAPPA_CSV)
    fam = pd.read_csv(FAMILIA_CSV)
    ref = referencia(fam)

    for modelo in sorted(kap.modelo.unique()):
        for bits in sorted(kap.bits.unique()):
            fig_percentiles(kap, modelo, bits)

    k = kap.join(ref, on=CLAVE)
    k["familia"] = "controlada"
    g = fam[fam.regimen == "gl_e"].join(ref, on=CLAVE)
    g["familia"] = "original"
    cols = ["bits", "familia", "kappa", "norma_wv", "norma_wo",
            "err_circuito", "e_0", "p_ref"]
    todo = pd.concat([k[cols], g[cols]], ignore_index=True)
    todo["producto"] = todo.norma_wv * todo.norma_wo / todo.p_ref
    todo["razon_e"] = todo.err_circuito / todo.e_0
    for bits in sorted(todo.bits.unique()):
        fig_colapso(todo, bits)

    hechas = sorted(p.name for p in SALIDA_DIR.glob("*kappa*.png"))
    hechas += sorted(p.name for p in SALIDA_DIR.glob("colapso*.png"))
    print("[figuras_kappa] generadas:")
    for n in hechas:
        print(f"  {n}")


if __name__ == "__main__":
    main()
