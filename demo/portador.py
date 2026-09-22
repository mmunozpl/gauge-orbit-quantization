"""carga del portador del sector valor-salida.

lo construye `scripts/extraer_sector_vo.py`. aquí solo se lee y se
verifica, porque el Space no lleva los modelos originales.

la regla de esta demo: no se reimplementa nada. el portador devuelve
los mismos tensores que `src.weights.w_v_w_o_*` entregaba al barrido, y
sobre ellos corren `src.gauges`, `src.quantizer` y `src.metrics` sin
un byte cambiado. si la demo y el paper discreparan, el culpable sería
el portador, y por eso lleva manifiesto con sha256.
"""

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

def _raiz() -> Path:
    """la raíz del árbol, igual en el repo que en el Space.

    returns:
        el directorio que contiene `src` y `demo`.
    """
    aqui = Path(__file__).resolve()
    for cand in (aqui.parent, *aqui.parents):
        if (cand / "src").is_dir() and (cand / "demo").is_dir():
            return cand
    return aqui.parent


RAIZ_DEFECTO = _raiz() / "artifacts" / "demo"


def _sha256(ruta: Path) -> str:
    """digest del fichero del portador.

    args:
        ruta: fichero a resumir.

    returns:
        sha256 en hexadecimal.
    """
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def carga(modelo: str, raiz: Path | None = None) -> dict:
    """lee el portador de un modelo y lo sube a float64.

    el fichero guarda float32, que es la dtype de origen; subirlo a
    float64 reproduce bit a bit lo que la extracción del barrido
    entregaba, y deja la aritmética de gauges en fp64.

    args:
        modelo: 'vitb' o 'pythia'.
        raiz: directorio del portador; por defecto artifacts/demo.

    returns:
        dict con 'w_v' [capas, cabezas, d_h, d], 'w_o' [capas,
        cabezas, d, d_h] en float64, y el manifiesto bajo 'meta'.

    raises:
        FileNotFoundError: si falta el portador o su manifiesto.
    """
    raiz = Path(raiz) if raiz is not None else RAIZ_DEFECTO
    fichero = raiz / f"sector_vo_{modelo}.safetensors"
    manif = raiz / f"sector_vo_{modelo}.json"
    if not fichero.exists() or not manif.exists():
        raise FileNotFoundError(
            f"falta el portador de {modelo} en {raiz}; constrúyelo con "
            f"scripts/extraer_sector_vo.py --modelo {modelo}")
    meta = json.loads(manif.read_text())
    t = load_file(str(fichero))
    return {"w_v": t["w_v"].double(), "w_o": t["w_o"].double(),
            "meta": meta}


def verifica_manifiesto(modelo: str, raiz: Path | None = None) -> None:
    """comprueba sha256 y formas contra el manifiesto.

    args:
        modelo: 'vitb' o 'pythia'.
        raiz: directorio del portador.

    raises:
        RuntimeError: si el digest o alguna forma no coinciden.
    """
    raiz = Path(raiz) if raiz is not None else RAIZ_DEFECTO
    fichero = raiz / f"sector_vo_{modelo}.safetensors"
    meta = json.loads((raiz / f"sector_vo_{modelo}.json").read_text())
    digest = _sha256(fichero)
    if digest != meta["sha256"]:
        raise RuntimeError(
            f"el portador de {modelo} no es el del manifiesto: "
            f"{digest[:16]} frente a {meta['sha256'][:16]}")
    t = load_file(str(fichero))
    for clave, forma in meta["formas"].items():
        if list(t[clave].shape) != forma:
            raise RuntimeError(
                f"forma inesperada en {clave}: {list(t[clave].shape)} "
                f"frente a {forma}")


def par(portador: dict, capa: int, cabeza: int) -> tuple:
    """(w_v, w_o) de una cabeza, en float64.

    args:
        portador: el dict que devuelve `carga`.
        capa: índice de capa.
        cabeza: índice de cabeza.

    returns:
        (w_v [d_h, d], w_o [d, d_h]) en float64.
    """
    return portador["w_v"][capa, cabeza], portador["w_o"][capa, cabeza]
