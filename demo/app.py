"""demo interactiva: la órbita de gauge valor-salida bajo cuantización.

tres pestañas. la órbita, donde se elige un punto y se ve qué le hace
la cuantización al circuito; el interruptor, con la validación
end-to-end ya medida; y qué se mide, con el método.

toda la aritmética la ejecuta el código del paper ---`src/gauges.py`,
`src/quantizer.py`, `src/metrics.py`--- sobre el portador ligero de
`demo/portador.py`. aquí no se reimplementa nada, y por eso
`scripts/paridad_demo.py` puede certificar la demo celda a celda
contra los csv publicados.
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib
import torch

matplotlib.use("Agg")               # sin servidor gráfico en el Space

from matplotlib.figure import Figure  # noqa: E402


def _raiz() -> Path:
    """la raíz del árbol, igual en el repo que en el Space.

    en el repositorio este fichero vive en `demo/`; en el Space, la
    app viaja a la raíz para que Gradio la encuentre. se sube hasta
    dar con el directorio que contiene `src` y `demo`, y así el mismo
    código sirve en los dos sitios sin condicionales.

    returns:
        el directorio raíz del árbol.
    """
    aqui = Path(__file__).resolve()
    for cand in (aqui.parent, *aqui.parents):
        if (cand / "src").is_dir() and (cand / "demo").is_dir():
            return cand
    return aqui.parent


sys.path.insert(0, str(_raiz()))

from demo.portador import carga, par  # noqa: E402
from src.gauges import (aplica_gauge, gauge_balanceado,  # noqa: E402
                        gl_condicionado, gl_general, ortogonal,
                        producto_normas)
from src.metrics import circuito, error_relativo  # noqa: E402
from src.quantizer import rtn_cuantiza_descuantiza  # noqa: E402

DOI = "https://doi.org/10.5281/zenodo.21630534"
REPO = "https://github.com/mmunozpl/gauge-orbit-quantization"
# gr.Markdown no compone matemáticas salvo que se le declaren los
# delimitadores; sin esto la prosa enseña los dólares en crudo.
LATEX = [{"left": "$$", "right": "$$", "display": True},
         {"left": "$", "right": "$", "display": False}]
# exponente publicado del ajuste conjunto sobre las dos familias
# (artifacts/tables/colapso_producto.csv, variable=producto, bits=4).
EXPONENTE = 0.9979
IDIOMAS = {"English": "en", "Español": "es"}
MODELOS = {"vitb": {"n_capas": 12, "n_cabezas": 12, "dim_cabeza": 64},
           "pythia": {"n_capas": 24, "n_cabezas": 16, "dim_cabeza": 64}}
# rejilla de fuerzas que la figura recorre en vivo, por familia
REJILLA = {"gl_escala": [32.0, 16.0, 8.0, 4.0, 2.0],
           "gl_kappa": [2.0, 10.0, 50.0, 200.0, 1000.0],
           "conforme": [0.1, 0.5, 1.0, 5.0, 50.0],
           "orto": [0.0, 1.0, 2.0, 3.0, 4.0]}
_CACHE: dict[str, dict] = {}

FAM = {
    "en": {"identidad": "identity (the trained gauge)",
           "conforme": "conformal orthogonal c·Q",
           "orto": "Haar orthogonal",
           "balanceado": "balanced point (constructed)",
           "gl_escala": "general GL, by scale",
           "gl_kappa": "general GL, by condition number κ"},
    "es": {"identidad": "identidad (el gauge entrenado)",
           "conforme": "conforme ortogonal c·Q",
           "orto": "ortogonal de Haar",
           "balanceado": "punto balanceado (construido)",
           "gl_escala": "GL general, por escala",
           "gl_kappa": "GL general, por número de condición κ"},
}
PARAM = {
    "en": {"identidad": "unused", "conforme": "coefficient c",
           "orto": "unused", "balanceado": "unused",
           "gl_escala": "scale (low = strong gauge)",
           "gl_kappa": "target κ"},
    "es": {"identidad": "sin uso", "conforme": "coeficiente c",
           "orto": "sin uso", "balanceado": "sin uso",
           "gl_escala": "escala (baja = gauge fuerte)",
           "gl_kappa": "κ objetivo"},
}
DEFECTO = {"identidad": 1.0, "conforme": 1.0, "orto": 1.0,
           "balanceado": 1.0, "gl_escala": 8.0, "gl_kappa": 50.0}


def portador(modelo: str) -> dict:
    """carga el portador de un modelo, con caché de proceso.

    args:
        modelo: 'vitb' o 'pythia'.

    returns:
        el dict de `demo.portador.carga`.
    """
    if modelo not in _CACHE:
        _CACHE[modelo] = carga(modelo)
    return _CACHE[modelo]


def construye_r(
    familia: str,
    parametro: float,
    dim_cabeza: int,
    gen: torch.Generator,
) -> torch.Tensor | None:
    """el gauge de una familia, en fp64.

    args:
        familia: clave de `FAM`.
        parametro: el mando de la familia; sin uso en varias.
        dim_cabeza: d_h.
        gen: generador de torch.

    returns:
        tensor [d_h, d_h] en float64, o None para la identidad.

    raises:
        ValueError: si la familia no existe.
    """
    if familia == "identidad":
        return None
    if familia == "conforme":
        return float(parametro) * ortogonal(dim_cabeza, gen)
    if familia == "orto":
        return ortogonal(dim_cabeza, gen)
    if familia == "gl_escala":
        return gl_general(dim_cabeza, float(parametro), gen)
    if familia == "gl_kappa":
        return gl_condicionado(dim_cabeza, float(parametro), gen)
    raise ValueError(f"familia desconocida: {familia}")


def mide_capa(
    modelo: str,
    capa: int,
    familia: str,
    parametro: float,
    semilla: int,
    bits: int,
) -> list[dict]:
    """mide todas las cabezas de una capa bajo un punto de la órbita.

    por cabeza devuelve p, d(R), cond(R), el error del circuito sin
    cuantizar ---el suelo de fp64, que la invariancia exige--- y el
    error tras cuantizar, con su cociente contra la identidad.

    args:
        modelo: 'vitb' o 'pythia'.
        capa: índice de capa.
        familia: clave de `FAM`.
        parametro: el mando de la familia.
        semilla: semilla del muestreo.
        bits: 4 u 8.

    returns:
        lista de dicts, una por cabeza.
    """
    info = MODELOS[modelo]
    port = portador(modelo)
    gen = torch.Generator().manual_seed(int(semilla))
    filas = []
    for h in range(info["n_cabezas"]):
        w_v, w_o = par(port, capa, h)
        m_ref = circuito(w_v, w_o)
        e_ident = error_relativo(rtn_cuantiza_descuantiza(w_v, bits),
                                 rtn_cuantiza_descuantiza(w_o, bits),
                                 m_ref)
        if familia == "balanceado":
            r, _ = gauge_balanceado(w_v, w_o)
        else:
            r = construye_r(familia, parametro, info["dim_cabeza"], gen)
        if r is None:
            a, b = w_v, w_o
        else:
            a, b = aplica_gauge(w_v, w_o, r)
        e_exacto = error_relativo(a, b, m_ref)
        e_quant = error_relativo(rtn_cuantiza_descuantiza(a, bits),
                                 rtn_cuantiza_descuantiza(b, bits),
                                 m_ref)
        if r is None:
            p, d_r, cond = 1.0, 0.0, 1.0
        else:
            p = producto_normas(w_v, w_o, r)
            s = torch.linalg.svdvals(r)
            d_r = float((s - s.mean()).norm() / s.norm())
            cond = float(s[0] / s[-1])
        filas.append({"cabeza": h, "p": p, "d_r": d_r, "cond": cond,
                      "e_exacto": e_exacto, "e_quant": e_quant,
                      "e_ident": e_ident,
                      "cociente": e_quant / e_ident})
    return filas


def _mediana(filas: list[dict], clave: str) -> float:
    """mediana de una columna de `mide_capa`."""
    v = sorted(f[clave] for f in filas)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def figura(
    modelo: str,
    capa: int,
    familia: str,
    semilla: int,
    bits: int,
    punto: tuple[float, float] | None,
    idi: str,
) -> Figure:
    """la ley del producto recorrida en vivo, con el punto elegido.

    se barre la rejilla de fuerzas de la familia sobre las cabezas de
    la capa y se dibuja $e/e_0$ contra $p$ en log-log, junto a la recta
    de exponente uno que el ajuste publicado da sobre 743 424 medidas.

    args:
        modelo: 'vitb' o 'pythia'.
        capa: índice de capa.
        familia: clave de `FAM`.
        semilla: semilla del muestreo.
        bits: 4 u 8.
        punto: (p, cociente) del punto elegido, o None.
        idi: 'en' o 'es'.

    returns:
        la figura de matplotlib.
    """
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.subplots()
    rejilla = REJILLA.get(familia, [DEFECTO[familia]])
    xs, ys = [], []
    for valor in rejilla:
        for f in mide_capa(modelo, capa, familia, valor, semilla, bits):
            if f["p"] > 0 and f["cociente"] > 0:
                xs.append(f["p"])
                ys.append(f["cociente"])
    if xs:
        ax.scatter(xs, ys, s=14, alpha=0.45,
                   label="barrido en vivo" if idi == "es"
                   else "live sweep")
        lo, hi = min(xs + [0.9]), max(xs + [1.1])
        rec = [lo, hi]
        ax.plot(rec, [x ** EXPONENTE for x in rec], color="crimson",
                linewidth=1.4,
                label=(f"ajuste publicado, exponente {EXPONENTE}"
                       if idi == "es"
                       else f"published fit, exponent {EXPONENTE}"))
    if punto is not None and punto[0] > 0 and punto[1] > 0:
        ax.scatter([punto[0]], [punto[1]], s=140, marker="*",
                   color="black", zorder=5,
                   label="tu punto" if idi == "es" else "your point")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$p(R)$")
    ax.set_ylabel("$e/e_0$")
    ax.set_title("La ley del producto" if idi == "es"
                 else "The product law")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


LEMA = {
 "en": ("### Same function before quantisation; different error "
        "after it\n\n"
        "$p(R)$ tracks the quantisation sensitivity of the "
        "value-output factorisation."),
 "es": ("### La misma función antes de cuantizar; distinto error "
        "después\n\n"
        "$p(R)$ mide la sensibilidad a la cuantización de la "
        "factorización valor-salida."),
}
# de dónde sale cada número: la demo mezcla medida en vivo sobre una
# capa con el ajuste publicado sobre el barrido entero, y quien mire
# tiene derecho a saber cuál es cuál sin abrir el código.
ALCANCE = {
 "en": ("**Live**, computed now over the chosen model, layer and "
        "point: the table above, the per-head table and the scattered "
        "points below. **Published**, measured offline over the full "
        "sweep of $743\\,424$ points across two models, two bit "
        "widths and two gauge families: the red line of exponent "
        f"{EXPONENTE}. A live sample of one layer is not the "
        "experiment; it is one slice of it."),
 "es": ("**En vivo**, calculado ahora sobre el modelo, la capa y el "
        "punto elegidos: la tabla de arriba, la tabla por cabeza y "
        "los puntos dispersos de abajo. **Publicado**, medido aparte "
        "sobre el barrido entero de $743\\,424$ puntos en dos "
        "modelos, dos anchuras y dos familias de gauge: la recta roja "
        f"de exponente {EXPONENTE}. Una muestra en vivo de una capa "
        "no es el experimento, es una rebanada suya."),
}

T = {
 "en": {
  "titulo": (
    "## Gauge equivalence does not survive quantisation\n\n"
    "The value-output pair of an attention head is not unique: for "
    "every invertible $R$, the substitution $W_v\\to R^\\top W_v$, "
    "$W_O\\to W_O R^{-\\top}$ leaves the OV circuit "
    "$M=W_O W_v$ **exactly** fixed. Pick a point of that orbit below "
    "and watch what quantisation does to it.\n\n"
    f"Paper and code: [repository]({REPO}) · previous work: "
    f"[{DOI.split('/')[-1]}]({DOI})"),
  "tab1": "The orbit", "tab2": "The switch", "tab3": "What is measured",
  "modelo": "Model", "capa": "Layer", "familia": "Gauge family",
  "param": "Parameter", "sem": "Seed", "bits": "Bit width",
  "btn": "Measure this point", "fig": "The product law",
  "alcance": "What is live and what is published",
  "porcabeza": "Per head", "res": "Result",
  "cab": ["head", "p(R)", "d(R)", "cond(R)", "e unquantised",
          f"e quantised", "e / e0"],
 },
 "es": {
  "titulo": (
    "## La equivalencia de gauge no sobrevive a la cuantización\n\n"
    "El par valor-salida de una cabeza de atención no es único: para "
    "toda $R$ invertible, la sustitución $W_v\\to R^\\top W_v$, "
    "$W_O\\to W_O R^{-\\top}$ deja el circuito OV $M=W_O W_v$ "
    "**exactamente** fijo. Elige abajo un punto de esa órbita y mira "
    "qué le hace la cuantización.\n\n"
    f"Artículo y código: [repositorio]({REPO}) · trabajo previo: "
    f"[{DOI.split('/')[-1]}]({DOI})"),
  "tab1": "La órbita", "tab2": "El interruptor", "tab3": "Qué se mide",
  "modelo": "Modelo", "capa": "Capa", "familia": "Familia de gauge",
  "param": "Parámetro", "sem": "Semilla", "bits": "Anchura",
  "btn": "Medir este punto", "fig": "La ley del producto",
  "alcance": "Qué es en vivo y qué es publicado",
  "porcabeza": "Por cabeza", "res": "Resultado",
  "cab": ["cabeza", "p(R)", "d(R)", "cond(R)", "e sin cuantizar",
          "e cuantizado", "e / e0"],
 },
}

QSM = {
 "en": (
  "### What is measured\n\n"
  "**The circuit error.** $e=\\lVert\\hat M-M\\rVert_F/"
  "\\lVert M\\rVert_F$, with $\\hat M$ recomposed in fp64 from the "
  "quantised pair. All gauge arithmetic is fp64.\n\n"
  "**The bound.** With a per-row quantiser bounded by $\\delta$, "
  "$e(R)\\le(\\delta_v+\\delta_O+\\delta_v\\delta_O)\\,g(R)$, and "
  "with $g$ the only gauge-dependent factor. Normalising it by its "
  "value at the identity gives $p=\\lVert W_OR^{-\\top}\\rVert_F"
  "\\lVert R^{\\top}W_v\\rVert_F/"
  "(\\lVert W_O\\rVert_F\\lVert W_v\\rVert_F)$. Two bounds do not "
  "divide: $e/e_0\\simeq p$ holds if the effective quantisation "
  "factor varies little across gauges. It is the hypothesis, not a "
  "corollary.\n\n"
  "**Two analytic ends.** The conformal orthogonal class $\\{cQ\\}$ "
  "gives $p\\equiv1$ for every $c$ and $Q$, so it is neutral for "
  "the bound. The "
  "balanced point minimises $p$ over the whole orbit, and the "
  "minimisers are that family up to a reciprocal rescaling.\n\n"
  "**$d(R)$** is the standard deviation of the singular values over "
  "their norm, the distance to $\\{cQ\\}$. It is exactly zero on the "
  "neutral class.\n\n"
  "**Nothing is reimplemented here.** The demo runs `src/gauges.py`, "
  "`src/quantizer.py` and `src/metrics.py` on a light carrier of the "
  "value-output sector, and `scripts/paridad_demo.py` certifies it "
  "cell by cell against the published sweeps.\n\n"
  "**What this demo does not do.** It does not run forwards. The "
  "end-to-end numbers in the second tab were measured offline over "
  "the 5000 images of the clean ImageNet-100 validation split."),
 "es": (
  "### Qué se mide\n\n"
  "**El error del circuito.** $e=\\lVert\\hat M-M\\rVert_F/"
  "\\lVert M\\rVert_F$, con $\\hat M$ recompuesto en fp64 desde el "
  "par cuantizado. Toda la aritmética de gauges es fp64.\n\n"
  "**La cota.** Con un cuantizador por fila acotado por $\\delta$, "
  "$e(R)\\le(\\delta_v+\\delta_O+\\delta_v\\delta_O)\\,g(R)$, y "
  "con $g$ el único factor que depende del gauge. Normalizarlo por "
  "su valor en la identidad da $p=\\lVert W_OR^{-\\top}\\rVert_F"
  "\\lVert R^{\\top}W_v\\rVert_F/"
  "(\\lVert W_O\\rVert_F\\lVert W_v\\rVert_F)$. Dos cotas no se "
  "dividen: $e/e_0\\simeq p$ vale si el factor efectivo de "
  "cuantización varía poco entre gauges. Es la hipótesis, no un "
  "corolario.\n\n"
  "**Dos extremos analíticos.** La clase conforme ortogonal "
  "$\\{cQ\\}$ da $p\\equiv1$ para toda $c$ y toda $Q$, así que es "
  "neutra para la cota. El punto balanceado minimiza $p$ sobre la "
  "órbita entera, "
  "y los mínimos son esa familia salvo un escalado recíproco.\n\n"
  "**$d(R)$** es la desviación típica de los valores singulares "
  "partida por su norma, la distancia a $\\{cQ\\}$. Vale cero "
  "exactamente sobre la clase neutra.\n\n"
  "**Aquí no se reimplementa nada.** La demo ejecuta "
  "`src/gauges.py`, `src/quantizer.py` y `src/metrics.py` sobre un "
  "portador ligero del sector valor-salida, y "
  "`scripts/paridad_demo.py` la certifica celda a celda contra los "
  "barridos publicados.\n\n"
  "**Qué no hace esta demo.** No corre forwards. Las cifras "
  "end-to-end de la segunda pestaña se midieron aparte sobre las "
  "5000 imágenes del val limpio de ImageNet-100."),
}


def tabla_e2e(idi: str) -> str:
    """la validación end-to-end medida, como markdown.

    args:
        idi: 'en' o 'es'.

    returns:
        el bloque de markdown de la pestaña del interruptor.
    """
    import csv

    ruta = _raiz() / "artifacts" / "tables" / "validacion_e2e.csv"
    if not ruta.exists():
        return "_sin tabla end-to-end_" if idi == "es" else "_no table_"
    filas = list(csv.DictReader(ruta.open()))
    nom = {"identidad_fp32": ("Identity, not quantised",
                              "Identidad, sin cuantizar"),
           "identidad": ("Identity, int4", "Identidad, int4"),
           "mejor_orto_muestreado": ("Best sampled orthogonal, int4",
                                     "Mejor ortogonal muestreado, int4"),
           "peor_orto_muestreado": ("Worst sampled orthogonal, int4",
                                    "Peor ortogonal muestreado, int4"),
           "balanceado": ("Balanced point, int4",
                          "Punto balanceado, int4"),
           "gl_escala2_fp32": ("GL scale 2, NOT quantised",
                               "GL escala 2, SIN cuantizar"),
           "gl_escala2": ("GL scale 2, int4", "GL escala 2, int4")}
    cab = ("| Condition | Top-1 |\n|---|---|"
           if idi == "en" else "| Condición | Top-1 |\n|---|---|")
    j = 0 if idi == "en" else 1
    cuerpo = "\n".join(
        f"| {nom.get(f['condicion'], (f['condicion'],) * 2)[j]} "
        f"| {float(f['top1']):.4f} |" for f in filas)
    intro = {
     "en": ("### The switch\n\nSeven forwards of ViT-B/16 over the "
            "5000 images of the clean ImageNet-100 validation split. "
            "The same gauge is measured twice, with and without "
            "quantisation. That pair of rows is the whole argument: "
            "the gauge leaves the function alone and quantisation "
            "destroys it. Unquantised, that gauge predicts the same "
            "class as the identity on **all 5000 images**, without a "
            "single discrepancy; quantised, it agrees on 44.\n\n"),
     "es": ("### El interruptor\n\nSiete forwards de ViT-B/16 sobre "
            "las 5000 imágenes del val limpio de ImageNet-100. El "
            "mismo gauge se mide dos veces, con y sin cuantizar. Ese "
            "par de filas es el argumento entero: el gauge deja la "
            "función quieta y la cuantización la destruye. Sin "
            "cuantizar, ese gauge predice la misma clase que la "
            "identidad en **las 5000 imágenes**, sin una sola "
            "discrepancia; cuantizado, coincide en 44.\n\n"),
    }
    return intro[idi] + cab + "\n" + cuerpo


def orbita(
    idioma: str,
    modelo: str,
    capa: int,
    familia: str,
    parametro: float,
    semilla: int,
    bits: int,
):
    """mide el punto elegido y compone resultado, tabla y figura.

    args:
        idioma: clave de `IDIOMAS`.
        modelo: 'vitb' o 'pythia'.
        capa: índice de capa.
        familia: clave de `FAM`.
        parametro: el mando de la familia.
        semilla: semilla del muestreo.
        bits: 4 u 8.

    returns:
        (markdown del resultado, filas de la tabla, figura).
    """
    idi = IDIOMAS[idioma]
    filas = mide_capa(modelo, int(capa), familia, parametro,
                      int(semilla), int(bits))
    p_med = _mediana(filas, "p")
    coc = _mediana(filas, "cociente")
    e_ex = max(f["e_exacto"] for f in filas)
    e_q = _mediana(filas, "e_quant")
    d_med = _mediana(filas, "d_r")
    cond_med = _mediana(filas, "cond")
    if idi == "es":
        res = (
          f"**En vivo** · {FAM['es'][familia]}, capa {int(capa)}, "
          f"int{int(bits)}. Medianas sobre las {len(filas)} cabezas "
          f"de esa capa.\n\n"
          f"| | |\n|---|---|\n"
          f"| $p(R)$ | {p_med:.4f} |\n"
          f"| $d(R)$, distancia a la clase neutra | {d_med:.4f} |\n"
          f"| $\\operatorname{{cond}}(R)$ | {cond_med:.3f} |\n"
          f"| error del circuito **sin cuantizar** (peor cabeza) "
          f"| {e_ex:.2e} |\n"
          f"| error del circuito en int{int(bits)} | {e_q:.4f} |\n"
          f"| $e/e_0$ medido | {coc:.4f} |\n"
          f"| $e/e_0$ que la cota predice | {p_med:.4f} |\n\n"
          f"Sin cuantizar el error se queda en el suelo de fp64: el "
          f"gauge no toca la función. Cuantizado, el mismo punto "
          f"multiplica el error por {coc:.2f}.")
    else:
        res = (
          f"**Live** · {FAM['en'][familia]}, layer {int(capa)}, "
          f"int{int(bits)}. Medians over the {len(filas)} heads of "
          f"that layer.\n\n"
          f"| | |\n|---|---|\n"
          f"| $p(R)$ | {p_med:.4f} |\n"
          f"| $d(R)$, distance to the neutral class | {d_med:.4f} |\n"
          f"| $\\operatorname{{cond}}(R)$ | {cond_med:.3f} |\n"
          f"| circuit error **unquantised** (worst head) "
          f"| {e_ex:.2e} |\n"
          f"| circuit error at int{int(bits)} | {e_q:.4f} |\n"
          f"| $e/e_0$ measured | {coc:.4f} |\n"
          f"| $e/e_0$ predicted by the bound | {p_med:.4f} |\n\n"
          f"Unquantised the error stays at the fp64 floor: the gauge "
          f"does not touch the function. Quantised, the same point "
          f"multiplies the error by {coc:.2f}.")
    tabla = [[f["cabeza"], round(f["p"], 4), round(f["d_r"], 4),
              round(f["cond"], 3), f"{f['e_exacto']:.2e}",
              round(f["e_quant"], 5), round(f["cociente"], 4)]
             for f in filas]
    fig = figura(modelo, int(capa), familia, int(semilla), int(bits),
                 (p_med, coc), idi)
    return res, tabla, fig


def capas_de(modelo: str):
    """actualiza el desplegable de capas al cambiar de modelo.

    args:
        modelo: 'vitb' o 'pythia'.

    returns:
        la actualización de gradio para el desplegable.
    """
    n = MODELOS[modelo]["n_capas"]
    return gr.update(choices=list(range(n)), value=min(5, n - 1))


def cambia_familia(idioma: str, familia: str):
    """reetiqueta el mando de la familia y le pone su valor por defecto.

    args:
        idioma: clave de `IDIOMAS`.
        familia: clave de `FAM`.

    returns:
        la actualización de gradio para el mando.
    """
    idi = IDIOMAS[idioma]
    return gr.update(label=PARAM[idi][familia], value=DEFECTO[familia])


def cambia_idioma(idioma: str, familia: str):
    """reetiqueta la interfaz entera.

    args:
        idioma: clave de `IDIOMAS`.
        familia: clave de `FAM` vigente.

    returns:
        la tupla de actualizaciones, en el orden del cableado.
    """
    idi = IDIOMAS[idioma]
    d = T[idi]
    fam_op = [(v, k) for k, v in FAM[idi].items()]
    return (gr.update(value=d["titulo"]),
            gr.update(value=LEMA[idi]),
            gr.update(label=d["modelo"]),
            gr.update(label=d["capa"]),
            gr.update(choices=fam_op, label=d["familia"]),
            gr.update(label=PARAM[idi][familia]),
            gr.update(label=d["sem"]),
            gr.update(label=d["bits"]),
            gr.update(value=d["btn"]),
            gr.update(label=d["res"]),
            gr.update(headers=d["cab"], label=d["porcabeza"]),
            gr.update(label=d["fig"]),
            gr.update(value=ALCANCE[idi], label=d["alcance"]),
            gr.update(value=tabla_e2e(idi)),
            gr.update(value=QSM[idi]),
            gr.update(label=d["tab1"]),
            gr.update(label=d["tab2"]),
            gr.update(label=d["tab3"]))


TITULO = ("Gauge equivalence does not survive quantisation - "
          "La equivalencia de gauge no sobrevive a la cuantización")

with gr.Blocks(title=TITULO) as demo:
    idi_o = gr.Radio(list(IDIOMAS), value="English",
                     label="Idioma / Language")
    cab = gr.Markdown(T["en"]["titulo"], latex_delimiters=LATEX)
    lema = gr.Markdown(LEMA["en"], latex_delimiters=LATEX)
    with gr.Tab(T["en"]["tab1"]) as pes1:
        with gr.Row():
            mod_o = gr.Dropdown(list(MODELOS), value="vitb",
                                label=T["en"]["modelo"])
            capa_o = gr.Dropdown(list(range(12)), value=5,
                                 label=T["en"]["capa"])
            fam_o = gr.Dropdown([(v, k) for k, v in FAM["en"].items()],
                                value="gl_escala",
                                label=T["en"]["familia"])
            par_o = gr.Number(value=8.0, label=PARAM["en"]["gl_escala"])
            sem_o = gr.Number(value=0, precision=0, label=T["en"]["sem"])
            bit_o = gr.Radio([4, 8], value=4, label=T["en"]["bits"])
        btn_o = gr.Button(T["en"]["btn"], variant="primary")
        res_o = gr.Markdown(label=T["en"]["res"], latex_delimiters=LATEX)
        fig_o = gr.Plot(label=T["en"]["fig"])
        alc_o = gr.Markdown(ALCANCE["en"], label=T["en"]["alcance"],
                            latex_delimiters=LATEX)
        tab_o = gr.Dataframe(headers=T["en"]["cab"],
                             label=T["en"]["porcabeza"])
        mod_o.change(capas_de, mod_o, capa_o)
        fam_o.change(cambia_familia, [idi_o, fam_o], par_o)
        btn_o.click(orbita,
                    [idi_o, mod_o, capa_o, fam_o, par_o, sem_o, bit_o],
                    [res_o, tab_o, fig_o])
    with gr.Tab(T["en"]["tab2"]) as pes2:
        e2e_o = gr.Markdown(tabla_e2e("en"), latex_delimiters=LATEX)
    with gr.Tab(T["en"]["tab3"]) as pes3:
        qsm_o = gr.Markdown(QSM["en"], latex_delimiters=LATEX)
    idi_o.change(cambia_idioma, [idi_o, fam_o],
                 [cab, lema, mod_o, capa_o, fam_o, par_o, sem_o,
                  bit_o, btn_o, res_o, tab_o, fig_o, alc_o, e2e_o,
                  qsm_o, pes1, pes2, pes3])

if __name__ == "__main__":
    demo.launch()
