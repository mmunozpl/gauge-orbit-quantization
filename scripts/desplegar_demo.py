"""monta el árbol que viaja al Space de Hugging Face.

el Space quiere `app.py` y `requirements.txt` en la raíz, más una
tarjeta `README.md` con su cabecera yaml. el árbol se monta aparte y
no se empuja desde aquí: el push lo decide el autor, y la política de
la casa es que todas las superficies cambien a la vez.

qué viaja:

* `app.py` (desde `demo/app.py`) y `demo/portador.py`
* `src/` entero, sin tocar, que es la propiedad que hace fiable la
  demo: ejecuta el mismo código que produjo las tablas
* los portadores de `artifacts/demo/` y la tabla end-to-end
* `requirements.txt` y la tarjeta del Space

qué NO viaja: los csv crudos, el paper, los documentos de proceso y
las rutas del árbol propio.

tras montarlo, la compuerta se corre CONTRA EL ÁRBOL MONTADO, que es
el que atenderá visitantes:

    python scripts/paridad_demo.py --raiz <destino>/artifacts/demo

el Space se llama `ManPla/gauge-orbit-quantization-demo`, en
correspondencia uno a uno con el repositorio del trabajo.

uso:
    python scripts/desplegar_demo.py --destino /ruta/al/space
"""

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TARJETA = """---
title: Gauge equivalence does not survive quantisation
emoji: "\\U0001FA9E"
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Gauge equivalence does not survive quantisation

Interactive companion to the note *Gauge equivalence does not survive
quantisation: value-output orbit, norm-product bound, and
consequences for rotational methods*.

Pick a point of the value-output gauge orbit and see what
quantisation does to the OV circuit. The gauge leaves the circuit
exactly fixed in real arithmetic; the demo shows the error staying at
the fp64 floor before quantising and exploding after it.

Nothing is reimplemented here: the app runs the note's own
`src/gauges.py`, `src/quantizer.py` and `src/metrics.py` on a light
carrier of the value-output sector. `scripts/paridad_demo.py`
certifies the demo cell by cell against the published sweeps.

The parity gate is not decoration: it replays the published sweeps'
own gauges and requires the demo to return the same numbers to
$3\\times10^{-15}$, against a tolerance of $10^{-12}$.

Code and paper: https://github.com/mmunozpl/gauge-orbit-quantization
Citable record: https://doi.org/10.5281/zenodo.22904207
Data behind every table and figure: https://huggingface.co/datasets/ManPla/gauge-orbit-quantization-results
"""


def copia(origen: Path, destino: Path) -> None:
    """copia fichero o árbol, creando el padre si hace falta."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if origen.is_dir():
        shutil.copytree(origen, destino, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copy2(origen, destino)


def main() -> None:
    """monta el árbol del Space en el destino indicado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", required=True)
    args = parser.parse_args()
    dst = Path(args.destino)

    faltan = [p for p in ("artifacts/demo/sector_vo_vitb.safetensors",
                          "artifacts/demo/sector_vo_pythia.safetensors",
                          "artifacts/tables/validacion_e2e.csv")
              if not (RAIZ / p).exists()]
    if faltan:
        raise SystemExit(
            "faltan piezas del despliegue: " + ", ".join(faltan) +
            "\nconstruye los portadores con scripts/extraer_sector_vo.py")

    dst.mkdir(parents=True, exist_ok=True)
    copia(RAIZ / "demo" / "app.py", dst / "app.py")
    copia(RAIZ / "demo" / "portador.py", dst / "demo" / "portador.py")
    copia(RAIZ / "demo" / "requirements.txt", dst / "requirements.txt")
    copia(RAIZ / "src", dst / "src")
    for pieza in ("sector_vo_vitb.safetensors", "sector_vo_vitb.json",
                  "sector_vo_pythia.safetensors",
                  "sector_vo_pythia.json"):
        copia(RAIZ / "artifacts" / "demo" / pieza,
              dst / "artifacts" / "demo" / pieza)
    copia(RAIZ / "artifacts" / "tables" / "validacion_e2e.csv",
          dst / "artifacts" / "tables" / "validacion_e2e.csv")
    (dst / "README.md").write_text(TARJETA)

    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    n = sum(1 for f in dst.rglob("*") if f.is_file())
    print(f"[desplegar] árbol montado en {dst}")
    print(f"[desplegar] {n} ficheros, {total / 1e6:.1f} MB")
    print(f"[desplegar] compuerta contra el árbol montado:")
    print(f"  python scripts/paridad_demo.py "
          f"--raiz {dst}/artifacts/demo")
    print("[desplegar] el push al Space lo decide el autor; aquí no "
          "se empuja nada.")


if __name__ == "__main__":
    main()
