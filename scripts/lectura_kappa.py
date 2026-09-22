"""lectura en frío del barrido de kappa, contra la regla preregistrada.

la regla se fijó por escrito antes de correr nada (ver el registro del
proyecto) y este script la aplica sin margen: suelo e_0 fijado y no
ajustado, ventana de ajuste definida por el error y no por décadas de
kappa, y veredicto ley/tendencia/sin-estructura por las tres
condiciones conjuntas.

uso:
    python scripts/lectura_kappa.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gauges import gl_condicionado, gl_general
from src.metrics import circuito, error_relativo
from src.quantizer import rtn_cuantiza_descuantiza

ENTRADA = "artifacts/logs/quant_kappa/quant_kappa.csv"
SALIDA_DIR = "artifacts/tables"

ALFA_PREDICHO = 1.0      # e1: predicción a priori del mecanismo
R2_MINIMO = 0.95
PUNTOS_MINIMOS = 8       # e2: sustituye a «>= 2 décadas»
TOL_ALFA_MODELOS = 0.15  # coincidencia entre dos arquitecturas, n=2
MEJORA_CODO = 0.02
RACHA_MAXIMA = 0.50      # fracción del rango; por encima, hay curvatura


def ajusta_potencia(kappa: np.ndarray, mediana: np.ndarray,
                    e_0: float) -> dict:
    """ajusta e = e_0 + a·(kappa-1)^alfa en log-log, con e_0 fijo.

    args:
        kappa: números de condición de las celdas de la ventana.
        mediana: mediana del error en cada celda.
        e_0: suelo medido en kappa=1, no ajustado.

    returns:
        dict con alfa, su error estándar, el intervalo al 95 %, r2, los
        residuos y la racha de signo más larga en fracción del rango.
    """
    x = np.log(kappa - 1.0)
    y = np.log(mediana - e_0)
    n = len(x)
    alfa, log_a = np.polyfit(x, y, 1)
    ajustado = alfa * x + log_a
    residuos = y - ajustado
    ss_res = float((residuos ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # error estándar de la pendiente en regresión simple
    se = float(np.sqrt(ss_res / (n - 2) / ((x - x.mean()) ** 2).sum()))
    # racha de residuos del mismo signo, en fracción de los puntos
    signos = np.sign(residuos)
    mejor = actual = 1
    for i in range(1, n):
        actual = actual + 1 if signos[i] == signos[i - 1] else 1
        mejor = max(mejor, actual)
    return {"alfa": float(alfa), "log_a": float(log_a), "se": se,
            "ic95": (float(alfa - 1.96 * se), float(alfa + 1.96 * se)),
            "r2": float(r2), "residuos": residuos,
            "racha": mejor / n, "n": n}


def ajusta_quebrada(kappa: np.ndarray, mediana: np.ndarray,
                    e_0: float) -> dict | None:
    """ajusta una ley quebrada de dos tramos y devuelve la mejor.

    args:
        kappa: números de condición de la ventana.
        mediana: mediana del error por celda.
        e_0: suelo medido, no ajustado.

    returns:
        dict con el codo, los dos exponentes y el r2 conjunto, o None
        si no hay puntos para dos tramos de al menos tres.
    """
    x = np.log(kappa - 1.0)
    y = np.log(mediana - e_0)
    n = len(x)
    if n < 6:
        return None
    ss_tot = float(((y - y.mean()) ** 2).sum())
    mejor = None
    for corte in range(3, n - 2):
        ss_res = 0.0
        alfas = []
        for tramo in (slice(0, corte), slice(corte - 1, n)):
            a, b = np.polyfit(x[tramo], y[tramo], 1)
            ss_res += float(((y[tramo] - (a * x[tramo] + b)) ** 2).sum())
            alfas.append(float(a))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if mejor is None or r2 > mejor["r2"]:
            mejor = {"codo": float(kappa[corte - 1]), "alfa_1": alfas[0],
                     "alfa_2": alfas[1], "r2": r2}
    return mejor


def consistencia_familia(dim_cabeza: int = 64, n_celdas: int = 24,
                         semilla: int = 12345) -> pd.DataFrame:
    """compara la familia controlada con la familia original a kappa igual.

    el barrido original no registró kappa, así que la comprobación usa
    muestras nuevas de la familia antigua (randn + escala·i), que son
    estadísticamente equivalentes a las que aquel consumió; queda
    declarado que no son las mismas muestras.

    args:
        dim_cabeza: d_h.
        n_celdas: número de pares (w_v, w_o) sintéticos a promediar.
        semilla: semilla del generador.

    returns:
        dataframe con kappa y error de cada muestra y familia.
    """
    gen = torch.Generator().manual_seed(semilla)
    filas = []
    for _ in range(n_celdas):
        w_v = torch.randn(dim_cabeza, 768, generator=gen,
                          dtype=torch.float64)
        w_o = torch.randn(768, dim_cabeza, generator=gen,
                          dtype=torch.float64)
        m_ref = circuito(w_v, w_o)
        for escala in (2.0, 8.0, 32.0):
            r = gl_general(dim_cabeza, escala, gen)
            k = float(torch.linalg.cond(r))
            filas.append({"familia": "original", "kappa": k,
                          "err": _err(w_v, w_o, r, m_ref)})
            r_c = gl_condicionado(dim_cabeza, k, gen)
            filas.append({"familia": "controlada", "kappa": k,
                          "err": _err(w_v, w_o, r_c, m_ref)})
    return pd.DataFrame(filas)


def _err(w_v, w_o, r, m_ref, bits: int = 4) -> float:
    """error del circuito bajo el gauge r, a la anchura dada."""
    r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
    return error_relativo(
        rtn_cuantiza_descuantiza(r.transpose(-2, -1) @ w_v, bits),
        rtn_cuantiza_descuantiza(w_o @ r_inv_t, bits), m_ref)


def lee_modelo(df: pd.DataFrame, modelo: str, bits: int) -> dict:
    """aplica la regla preregistrada a un modelo y una anchura.

    args:
        df: barrido completo.
        modelo: 'pythia' o 'vitb'.
        bits: 4 u 8.

    returns:
        dict con el suelo, la ventana, el ajuste y el veredicto.
    """
    sub = df[(df.modelo == modelo) & (df.bits == bits)]
    por_kappa = sub.groupby("kappa").err_circuito.median().sort_index()
    e_0 = float(por_kappa.loc[por_kappa.index.min()])
    dentro = por_kappa[(por_kappa >= 2 * e_0) & (por_kappa <= 1.0)]
    res = {"modelo": modelo, "bits": bits, "e_0": e_0,
           "n_ventana": len(dentro),
           "kappa_min": float(dentro.index.min()) if len(dentro) else None,
           "kappa_max": float(dentro.index.max()) if len(dentro) else None,
           "monotona": bool(por_kappa.is_monotonic_increasing)}
    if len(dentro) < PUNTOS_MINIMOS:
        res["veredicto"] = ("sin_estructura" if not res["monotona"]
                            else "tendencia")
        res["motivo"] = (f"solo {len(dentro)} puntos en la ventana "
                         f"(mínimo {PUNTOS_MINIMOS})")
        return res
    ajuste = ajusta_potencia(dentro.index.values, dentro.values, e_0)
    quebrada = ajusta_quebrada(dentro.index.values, dentro.values, e_0)
    res.update({k: v for k, v in ajuste.items() if k != "residuos"})
    res["quebrada"] = quebrada
    return res


def main() -> None:
    """lee el barrido en frío y escribe el veredicto y sus tablas."""
    df = pd.read_csv(ENTRADA)
    print(f"=== barrido de kappa: {len(df)} filas, "
          f"{df.modelo.nunique()} modelos ===\n")

    resultados = []
    for modelo in sorted(df.modelo.unique()):
        for bits in sorted(df.bits.unique()):
            r = lee_modelo(df, modelo, bits)
            resultados.append(r)
            print(f"--- {modelo} int{bits} ---")
            print(f"  e_0 (kappa=1, medido, no ajustado) = {r['e_0']:.5f}")
            print(f"  ventana [2·e_0, 1]: {r['n_ventana']} puntos"
                  + (f", kappa {r['kappa_min']:.2f}–{r['kappa_max']:.2f}"
                     if r["n_ventana"] else ""))
            if "alfa" in r:
                print(f"  alfa = {r['alfa']:.3f}  "
                      f"IC95 [{r['ic95'][0]:.3f}, {r['ic95'][1]:.3f}]  "
                      f"(predicho {ALFA_PREDICHO})")
                print(f"  R2 = {r['r2']:.4f}   racha de signo = "
                      f"{r['racha']:.0%} del rango")
                if r["quebrada"]:
                    q = r["quebrada"]
                    print(f"  quebrada: codo={q['codo']:.1f}, "
                          f"alfa1={q['alfa_1']:.3f}, "
                          f"alfa2={q['alfa_2']:.3f}, R2={q['r2']:.4f} "
                          f"(mejora {q['r2'] - r['r2']:+.4f})")
            else:
                print(f"  veredicto anticipado: {r['veredicto']} "
                      f"({r['motivo']})")
            print()

    # veredicto conjunto por anchura: exige las tres condiciones
    print("=== VEREDICTO contra la regla preregistrada ===")
    for bits in sorted(df.bits.unique()):
        pareja = [r for r in resultados if r["bits"] == bits]
        if not all("alfa" in r for r in pareja):
            print(f"  int{bits}: "
                  + ", ".join(f"{r['modelo']}={r.get('veredicto','?')}"
                              for r in pareja))
            continue
        alfas = [r["alfa"] for r in pareja]
        dif = abs(alfas[0] - alfas[1]) / max(abs(a) for a in alfas)
        cond = {
            f"R2 >= {R2_MINIMO}": all(r["r2"] >= R2_MINIMO for r in pareja),
            f">= {PUNTOS_MINIMOS} puntos":
                all(r["n_ventana"] >= PUNTOS_MINIMOS for r in pareja),
            f"alfa coincide (±{TOL_ALFA_MODELOS:.0%})":
                dif <= TOL_ALFA_MODELOS,
            f"sin curvatura (racha <= {RACHA_MAXIMA:.0%})":
                all(r["racha"] <= RACHA_MAXIMA for r in pareja),
        }
        for nombre, ok in cond.items():
            print(f"  int{bits}  {'✓' if ok else '✗'}  {nombre}")
        veredicto = "LEY" if all(cond.values()) else "TENDENCIA"
        print(f"  int{bits}  -> {veredicto}   "
              f"(alfas {alfas[0]:.3f} / {alfas[1]:.3f}, "
              f"difieren {dif:.1%})\n")

    print("=== consistencia: familia controlada vs original, "
          "a kappa igualado ===")
    cons = consistencia_familia()
    for fam, g in cons.groupby("familia"):
        print(f"  {fam:>11}: n={len(g)}, err mediano={g.err.median():.4f}")
    emparejado = cons.pivot_table(index=cons.groupby("familia").cumcount(),
                                  columns="familia", values="err")
    razon = (emparejado["controlada"] / emparejado["original"]).median()
    print(f"  razón mediana controlada/original = {razon:.3f} "
          f"(1,0 = la ley es de kappa, no de la familia)")

    salida = Path(SALIDA_DIR) / "lectura_kappa.csv"
    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ("quebrada",)} for r in resultados]
                 ).to_csv(salida, index=False)
    cons.to_csv(Path(SALIDA_DIR) / "consistencia_kappa.csv", index=False)
    print(f"\n[lectura_kappa] tablas -> {salida} y consistencia_kappa.csv")


if __name__ == "__main__":
    main()
