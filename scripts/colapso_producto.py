"""contraste del colapso entre familias, sobre el barrido completo.

la derivación de forma cerrada (ver el registro) predice, sin ajustar
nada, que el error relativo del circuito escala con el producto de las
normas de los factores transformados:

    e/e_0 ≈ (‖W_v R‖·‖R⁻¹W_O‖)/(‖W_v‖·‖W_O‖) ≡ p,

es decir exponente 1. la afirmación es *entre familias*: dentro de la
familia controlada p es función determinista de kappa, así que solo
comparar familias distintas al mismo p la pone a prueba.

la referencia común de ambas familias es la identidad de cada celda,
registrada por scripts/sweep_familia.py.

uso:
    python scripts/colapso_producto.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KAPPA_CSV = "artifacts/logs/quant_kappa/quant_kappa.csv"
FAMILIA_CSV = "artifacts/logs/quant_kappa/quant_familia.csv"
SALIDA = "artifacts/tables/colapso_producto.csv"

CLAVE = ["modelo", "capa", "cabeza", "bits"]
EXPONENTE_PREDICHO = 1.0
TOL_EXPONENTE = (0.97, 1.03)
TOL_SEPARACION = 0.02
R2_MINIMO = 0.99


def referencia(fam: pd.DataFrame) -> pd.DataFrame:
    """extrae e_0 y el producto de normas de la identidad por celda.

    args:
        fam: barrido de la familia original, con filas de identidad.

    returns:
        dataframe indexado por celda con e_0 y p_ref.
    """
    ident = fam[fam.regimen == "identidad"].copy()
    ident["p_ref"] = ident.norma_wv * ident.norma_wo
    return (ident.set_index(CLAVE)[["err_circuito", "p_ref"]]
            .rename(columns={"err_circuito": "e_0"}))


def ajusta(sub: pd.DataFrame, var: str) -> dict:
    """ajusta log(e/e_0) contra log(var) y mide la separación de familias.

    args:
        sub: filas con columnas razon_e, familia y la variable.
        var: 'producto' o 'kappa'.

    returns:
        dict con exponente, r2 y separación entre familias en log.
    """
    x = np.log(sub[var].values)
    y = np.log(sub.razon_e.values)
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    r2 = 1.0 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    sep = abs(res[sub.familia.values == "original"].mean()
              - res[sub.familia.values == "controlada"].mean())
    return {"exponente": float(a), "r2": float(r2),
            "separacion": float(sep), "n": len(sub)}


def main() -> None:
    """contrasta la predicción sobre el conjunto completo."""
    kap = pd.read_csv(KAPPA_CSV)
    fam = pd.read_csv(FAMILIA_CSV)
    ref = referencia(fam)

    kap = kap.join(ref, on=CLAVE)
    kap["familia"] = "controlada"
    gl = fam[fam.regimen == "gl_e"].join(ref, on=CLAVE)
    gl["familia"] = "original"

    cols = ["modelo", "bits", "familia", "kappa", "norma_wv", "norma_wo",
            "err_circuito", "e_0", "p_ref"]
    todo = pd.concat([kap[cols], gl[cols]], ignore_index=True)
    todo["producto"] = todo.norma_wv * todo.norma_wo / todo.p_ref
    todo["razon_e"] = todo.err_circuito / todo.e_0
    # kappa=1 no entra en el ajuste en kappa (log 1 = 0 es el ancla,
    # no aporta pendiente) pero sí en el del producto.
    print(f"=== colapso sobre el conjunto completo: {len(todo)} filas "
          f"({todo.familia.value_counts().to_dict()}) ===\n")

    filas = []
    for bits in sorted(todo.bits.unique()):
        sub = todo[todo.bits == bits]
        print(f"--- int{bits} (n={len(sub)}) ---")
        for var in ("kappa", "producto"):
            s = sub[sub[var] > 1.0] if var == "kappa" else sub
            r = ajusta(s, var)
            r.update({"bits": bits, "variable": var})
            filas.append(r)
            print(f"  e/e_0 ~ {var:>9}^a :  a={r['exponente']:.4f}   "
                  f"R2={r['r2']:.5f}   separación entre familias="
                  f"{r['separacion']:.4f}")
        print()

    print("=== CONTRASTE contra la predicción escrita antes ===")
    for bits in sorted(todo.bits.unique()):
        r = next(f for f in filas
                 if f["bits"] == bits and f["variable"] == "producto")
        cond = {
            f"exponente en [{TOL_EXPONENTE[0]}, {TOL_EXPONENTE[1]}]":
                TOL_EXPONENTE[0] <= r["exponente"] <= TOL_EXPONENTE[1],
            f"separación < {TOL_SEPARACION}":
                r["separacion"] < TOL_SEPARACION,
            f"R2 >= {R2_MINIMO}": r["r2"] >= R2_MINIMO,
        }
        for nombre, ok in cond.items():
            print(f"  int{bits}  {'✓' if ok else '✗'}  {nombre}")
        fallo = "PREDICCIÓN CONFIRMADA" if all(cond.values()) else \
            "NO CONFIRMADA"
        print(f"  int{bits}  -> {fallo}  "
              f"(exponente {r['exponente']:.4f}, predicho "
              f"{EXPONENTE_PREDICHO})\n")

    print("=== descriptivo: mismo ajuste restringido a e <= 1 ===")
    for bits in sorted(todo.bits.unique()):
        sub = todo[(todo.bits == bits) & (todo.err_circuito <= 1.0)]
        r = ajusta(sub, "producto")
        print(f"  int{bits}: a={r['exponente']:.4f}  R2={r['r2']:.5f}  "
              f"separación={r['separacion']:.4f}  n={r['n']}")

    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    print(f"\n[colapso] tabla -> {SALIDA}")


if __name__ == "__main__":
    main()
