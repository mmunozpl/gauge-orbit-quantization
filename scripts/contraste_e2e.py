"""contraste pareado de la validación end-to-end.

las condiciones del end-to-end se evalúan sobre exactamente las
mismas 5000 imágenes y en el mismo orden, así que la comparación
admite un contraste pareado y no necesita apoyarse en un «ruido de
evaluación» declarado de palabra. dos estadísticos, ambos sobre el
vector de aciertos por imagen que `validacion_e2e.py` guarda:

- mcnemar exacto sobre los discordantes (b = acierta a y falla b,
  c = al revés): prueba binomial de dos colas con p = 0,5, que es la
  prueba correcta para dos clasificadores sobre la misma muestra.
- bootstrap pareado de imágenes (10 000 remuestreos, semilla fija)
  para el intervalo de confianza del 95 % de la diferencia de top-1.
- acuerdo de clase predicha, cuando el fichero de predicciones está.
  cero discordantes de mcnemar dice que dos condiciones aciertan y
  fallan las MISMAS imágenes, y deja abierto que fallen prediciendo
  clases distintas. afirmar «predice lo mismo» exige comparar el
  argmax, y esa columna es la que lo respalda.

la lectura se fija antes de mirar: dos condiciones son
indistinguibles si el intervalo del bootstrap contiene el cero y
mcnemar no baja de 0,05.

uso:
    python scripts/contraste_e2e.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENTRADA = "artifacts/logs/quant_orbita/aciertos_e2e.pt"
ENTRADA_PRED = "artifacts/logs/quant_orbita/predicciones_e2e.pt"
SALIDA = "artifacts/tables/contraste_e2e.csv"
N_BOOTSTRAP = 10_000
SEMILLA = 0
ALFA = 0.05

# los pares que la nota afirma, cada uno con la afirmación que
# sostiene. se fijan aquí para que el contraste no se elija después
# de ver los números.
PARES = [
    ("mejor_orto_muestreado", "peor_orto_muestreado",
     "q1 end-to-end: la orbita ortogonal es plana"),
    ("identidad", "mejor_orto_muestreado",
     "la identidad no es un punto especial dentro del subgrupo"),
    ("identidad", "peor_orto_muestreado",
     "la identidad no es un punto especial dentro del subgrupo"),
    ("identidad", "balanceado",
     "el punto balanceado, gauge no ortogonal, no es toxico"),
    ("identidad_fp32", "identidad",
     "coste de cuantizar int4 sin tocar el gauge"),
    ("identidad_fp32", "gl_escala2_fp32",
     "control: el gauge gl solo, sin cuantizar"),
    ("identidad", "gl_escala2",
     "la cola toxica, end-to-end"),
]


def mcnemar_exacto(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """mcnemar exacto (binomial de dos colas) sobre los discordantes.

    args:
        a: vector bool de aciertos de la primera condición.
        b: vector bool de aciertos de la segunda.

    returns:
        (b_solo_a, c_solo_b, p_valor); p = 1,0 si no hay discordantes.
    """
    solo_a = int(np.sum(a & ~b))
    solo_b = int(np.sum(~a & b))
    n = solo_a + solo_b
    if n == 0:
        return solo_a, solo_b, 1.0
    p = float(stats.binomtest(solo_a, n, 0.5).pvalue)
    return solo_a, solo_b, p


def bootstrap_pareado(
    a: np.ndarray,
    b: np.ndarray,
    n_rep: int = N_BOOTSTRAP,
    semilla: int = SEMILLA,
) -> tuple[float, float]:
    """intervalo percentil del 95 % de la diferencia de top-1.

    se remuestrean ÍNDICES DE IMAGEN, no condiciones: el pareado se
    conserva porque las dos condiciones ven la misma imagen en cada
    remuestreo.

    args:
        a: vector bool de aciertos de la primera condición.
        b: vector bool de aciertos de la segunda.
        n_rep: remuestreos.
        semilla: semilla del generador.

    returns:
        (extremo inferior, extremo superior) de la diferencia a - b.
    """
    rng = np.random.default_rng(semilla)
    dif = a.astype(np.float64) - b.astype(np.float64)
    idx = rng.integers(0, len(dif), size=(n_rep, len(dif)))
    medias = dif[idx].mean(axis=1)
    return (float(np.percentile(medias, 100 * ALFA / 2)),
            float(np.percentile(medias, 100 * (1 - ALFA / 2))))


def main() -> None:
    """corre los contrastes preespecificados y los vuelca a tabla."""
    entrada = Path(ENTRADA)
    if not entrada.exists():
        raise SystemExit(
            f"faltan los aciertos por imagen en {entrada}; "
            f"corre antes scripts/validacion_e2e.py")
    aciertos = {k: v.numpy().astype(bool)
                for k, v in torch.load(entrada).items()}
    ruta_pred = Path(ENTRADA_PRED)
    predichas = ({k: v.numpy() for k, v in torch.load(ruta_pred).items()}
                 if ruta_pred.exists() else {})
    if not predichas:
        print("[contraste] sin fichero de clases predichas; la columna "
              "de acuerdo de argmax queda vacía")

    filas = []
    for nombre_a, nombre_b, afirmacion in PARES:
        if nombre_a not in aciertos or nombre_b not in aciertos:
            print(f"[contraste] falta {nombre_a} o {nombre_b}, se salta")
            continue
        a, b = aciertos[nombre_a], aciertos[nombre_b]
        solo_a, solo_b, p_val = mcnemar_exacto(a, b)
        lo, hi = bootstrap_pareado(a, b)
        dif = float(a.mean() - b.mean())
        indistinguible = lo <= 0.0 <= hi and p_val >= ALFA
        acuerdo = ""
        if nombre_a in predichas and nombre_b in predichas:
            acuerdo = float(
                (predichas[nombre_a] == predichas[nombre_b]).mean())
        filas.append({
            "condicion_a": nombre_a, "condicion_b": nombre_b,
            "top1_a": float(a.mean()), "top1_b": float(b.mean()),
            "dif_pp": 100 * dif,
            "ic95_inf_pp": 100 * lo, "ic95_sup_pp": 100 * hi,
            "discordantes_a": solo_a, "discordantes_b": solo_b,
            "mcnemar_p": p_val,
            "indistinguibles": indistinguible,
            "acuerdo_argmax": acuerdo,
            "afirmacion": afirmacion})

    tabla = pd.DataFrame(filas)
    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(out, index=False)

    print(f"\n=== contraste pareado — {len(aciertos[PARES[0][0]])} "
          f"imágenes, {N_BOOTSTRAP} remuestreos ===\n")
    for f in filas:
        marca = "indistinguibles" if f["indistinguibles"] else "distintas"
        print(f"{f['condicion_a']} vs {f['condicion_b']}")
        print(f"  top1 {f['top1_a']:.4f} vs {f['top1_b']:.4f}  "
              f"dif {f['dif_pp']:+.2f} pp  "
              f"ic95 [{f['ic95_inf_pp']:+.2f}, {f['ic95_sup_pp']:+.2f}] pp")
        print(f"  discordantes {f['discordantes_a']}/{f['discordantes_b']}  "
              f"mcnemar p = {f['mcnemar_p']:.4f}  -> {marca}")
        if f["acuerdo_argmax"] != "":
            n_img = len(aciertos[f["condicion_a"]])
            print(f"  acuerdo de clase predicha: "
                  f"{f['acuerdo_argmax']:.4f} "
                  f"({int(round(f['acuerdo_argmax'] * n_img))}/{n_img})")
        print(f"  ({f['afirmacion']})\n")
    print(f"[guardado] {out}")


if __name__ == "__main__":
    main()
