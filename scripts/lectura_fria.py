"""lectura en frío del barrido: q1/q2/q3 y la cola gl, contra los
listones preregistrados (spec sec. 3). no se ejecuta hasta que el
csv del modelo esté completo —lectura en frío por modelo, no
incremental por capa—.

uso:
    python scripts/lectura_fria.py --modelo pythia
    python scripts/lectura_fria.py --modelo vitb
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENTRADA = "artifacts/logs/quant_orbita/quant_orbita.csv"
SALIDA = "artifacts/tables/lectura_fria_{modelo}.csv"


def q1_anchura(df: pd.DataFrame) -> pd.DataFrame:
    """rango ortogonal (orto_ph ∪ orto_comp) contra el error identidad.

    por (capa, cabeza, bits): ratio = peor_ortogonal / identidad.
    material si >= 2, estrecho si < 1.25, si no entre medias.
    """
    ident = df[df.regimen == "identidad"].set_index(
        ["capa", "cabeza", "bits"]).err_circuito
    orto = df[df.regimen.isin(["orto_ph", "orto_comp"])]
    peor = orto.groupby(["capa", "cabeza", "bits"]).err_circuito.max()
    ratio = (peor / ident).rename("ratio_q1").reset_index()
    ratio["veredicto"] = ratio.ratio_q1.apply(
        lambda r: "material" if r >= 2.0
        else ("estrecho" if r < 1.25 else "entremedias"))
    return ratio


def q2_gauge_entrenado(df: pd.DataFrame) -> pd.DataFrame:
    """identidad contra el mejor muestreado (orto_ph/orto_comp/gl_e).

    por (capa, cabeza, bits): ratio = identidad / mejor_muestreado.
    lejos del suelo si > 1.5, cerca si < 1.1.
    """
    ident = df[df.regimen == "identidad"].set_index(
        ["capa", "cabeza", "bits"]).err_circuito
    muestreado = df[df.regimen != "identidad"]
    mejor = muestreado.groupby(
        ["capa", "cabeza", "bits"]).err_circuito.min()
    ratio = (ident / mejor).rename("ratio_q2").reset_index()
    ratio["veredicto"] = ratio.ratio_q2.apply(
        lambda r: "lejos_del_suelo" if r > 1.5
        else ("cerca" if r < 1.1 else "entremedias"))
    return ratio


def q3_por_cabeza_vs_compartida(df: pd.DataFrame) -> pd.DataFrame:
    """mejor-de-k por-cabeza contra mejor-de-k compartida.

    por (capa, cabeza, bits): mejora relativa =
    (comp_mejor - ph_mejor) / comp_mejor. material si mediana >= 0.20,
    nulo si mediana < 0.05.
    """
    ph = df[df.regimen == "orto_ph"].groupby(
        ["capa", "cabeza", "bits"]).err_circuito.min().rename("ph_mejor")
    comp = df[df.regimen == "orto_comp"].groupby(
        ["capa", "cabeza", "bits"]).err_circuito.min().rename("comp_mejor")
    tabla = pd.concat([ph, comp], axis=1).reset_index()
    tabla["mejora_relativa"] = (
        (tabla.comp_mejor - tabla.ph_mejor) / tabla.comp_mejor)
    return tabla


def cola_gl(df: pd.DataFrame) -> pd.DataFrame:
    """mediana de error gl_e por escala: ¿degrada la escala baja?"""
    gl = df[df.regimen == "gl_e"]
    return gl.groupby(["escala", "bits"]).err_circuito.agg(
        ["median", "mean", "max"]).reset_index()


def main() -> None:
    """corre las cuatro lecturas y las vuelca a artifacts/tables/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True)
    args = parser.parse_args()

    df = pd.read_csv(ENTRADA)
    df = df[df.modelo == args.modelo]
    if df.empty:
        raise SystemExit(f"sin filas para modelo={args.modelo} en "
                         f"{ENTRADA}; ¿corriste el barrido?")

    q1 = q1_anchura(df)
    q2 = q2_gauge_entrenado(df)
    q3 = q3_por_cabeza_vs_compartida(df)
    gl = cola_gl(df)

    out_dir = Path(f"artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in [("q1", q1), ("q2", q2), ("q3", q3),
                          ("gl", gl)]:
        tabla.to_csv(out_dir / f"lectura_fria_{args.modelo}_{nombre}.csv",
                     index=False)

    print(f"\n=== lectura en frío — {args.modelo} ({len(df)} filas) ===")
    print("\nQ1 — anchura de la órbita (por capa/cabeza/bits):")
    print(q1.veredicto.value_counts())
    print(f"mediana ratio_q1 = {q1.ratio_q1.median():.2f}x  "
         f"(rango [{q1.ratio_q1.min():.2f}, {q1.ratio_q1.max():.2f}])")

    print("\nQ2 — el gauge entrenado (por capa/cabeza/bits):")
    print(q2.veredicto.value_counts())
    print(f"mediana ratio_q2 = {q2.ratio_q2.median():.2f}x  "
         f"(rango [{q2.ratio_q2.min():.2f}, {q2.ratio_q2.max():.2f}])")

    mediana_q3 = q3.mejora_relativa.median()
    veredicto_q3 = ("material" if mediana_q3 >= 0.20
                    else ("nulo" if mediana_q3 < 0.05 else "entremedias"))
    print(f"\nQ3 — por-cabeza vs compartida: mediana mejora relativa "
         f"= {mediana_q3:.1%} -> {veredicto_q3}")

    print("\nCola gl_e por escala (mediana/media/máx del error):")
    print(gl.to_string(index=False))

    print(f"\n[guardado] tablas en {out_dir}/lectura_fria_{args.modelo}_*.csv")


if __name__ == "__main__":
    main()
