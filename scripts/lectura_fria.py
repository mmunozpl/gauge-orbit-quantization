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
    """mejor-de-k por-cabeza contra la mejor rotación compartida.

    corrección metodológica del 22-09-2026. la primera implementación
    tomaba, dentro del régimen compartido, el mínimo por cabeza sobre
    las k muestras: cada cabeza acababa escogiendo un índice distinto,
    y el brazo «compartido» dejaba de serlo. el contraste comparaba
    entonces libertad por-cabeza contra libertad por-cabeza, y daba
    una mejora nula por construcción.

    la lectura corregida escoge UNA sola r por capa —la muestra que
    minimiza el error medio sobre las cabezas de esa capa, que es el
    objetivo agregado que un método rotacional optimizaría— y la
    evalúa cabeza a cabeza. la pregunta q3 y su listón son los
    preregistrados; la implementación de «compartida» es lo que se
    corrige.

    por (capa, cabeza, bits): mejora relativa =
    (comp_unico - ph_mejor) / comp_unico. material si la mediana es
    >= 0.20, nulo si < 0.05.
    """
    ph = df[df.regimen == "orto_ph"].groupby(
        ["capa", "cabeza", "bits"]).err_circuito.min().rename("ph_mejor")
    comp = df[df.regimen == "orto_comp"]
    # una sola muestra por (capa, bits): la de menor error medio sobre
    # las cabezas de la capa.
    medio = comp.groupby(
        ["capa", "bits", "muestra"]).err_circuito.mean().reset_index()
    elegida = medio.loc[
        medio.groupby(["capa", "bits"]).err_circuito.idxmin(),
        ["capa", "bits", "muestra"]]
    comp_unico = comp.merge(elegida, on=["capa", "bits", "muestra"]).set_index(
        ["capa", "cabeza", "bits"]).err_circuito.rename("comp_unico")
    # la lectura vieja, conservada como línea de regresión: hace
    # legible la corrección en vez de sustituirla en silencio.
    comp_porcabeza = comp.groupby(
        ["capa", "cabeza", "bits"]).err_circuito.min().rename(
            "comp_min_por_cabeza")
    tabla = pd.concat(
        [ph, comp_unico, comp_porcabeza], axis=1).reset_index()
    tabla["muestra_compartida"] = tabla.merge(
        elegida, on=["capa", "bits"], how="left").muestra.values
    tabla["mejora_relativa"] = (
        (tabla.comp_unico - tabla.ph_mejor) / tabla.comp_unico)
    tabla["mejora_implementacion_vieja"] = (
        (tabla.comp_min_por_cabeza - tabla.ph_mejor)
        / tabla.comp_min_por_cabeza)
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
    print(f"\nQ3 — por-cabeza vs compartida (una r por capa): mediana "
         f"mejora relativa = {mediana_q3:.2%} -> {veredicto_q3}")
    print(f"     percentiles 10/90: "
         f"{q3.mejora_relativa.quantile(0.10):.2%} / "
         f"{q3.mejora_relativa.quantile(0.90):.2%}")
    print(f"     [regresión] la implementación vieja, que dejaba a cada "
         f"cabeza elegir su índice dentro del brazo compartido, daba "
         f"{q3.mejora_implementacion_vieja.median():.2%}")

    print("\nCola gl_e por escala (mediana/media/máx del error):")
    print(gl.to_string(index=False))

    print(f"\n[guardado] tablas en {out_dir}/lectura_fria_{args.modelo}_*.csv")


if __name__ == "__main__":
    main()
