"""sella el estado local antes de publicar: manifiesto de bytes.

la disciplina de la casa es identidad documental: los mismos bytes en
todas partes. este script escribe el manifiesto que despues se
contrasta contra lo publicado, y no sube nada.

el orden importa y la leccion viene del trabajo previo. alli se planto
un tag afirmando identidad con el deposito y el zip depositado resulto
ser el arbol de seis commits antes, asi que la verificacion se hace
DESPUES de depositar, descargando el zip y comparandolo contra este
manifiesto.

uso:
    python scripts/sellar.py                 # escribe el manifiesto
    python scripts/sellar.py --verificar     # lo contrasta con el disco
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "artifacts" / "sello.json"
# las piezas cuya identidad se afirma en algun sitio: el PDF que el
# tag congela y el deposito lleva, y las tablas de las que sale cada
# cifra del texto.
PIEZAS = [
    "paper/gauge_orbit_quantization_es.pdf",
    "paper/gauge_orbit_quantization_en.pdf",
    "CITATION.cff",
    "README.md",
    "LEEME.md",
]
PATRONES = ["artifacts/tables/*.csv", "artifacts/demo/*.safetensors"]


def sha256(ruta: Path) -> str:
    """digest de un fichero.

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


def piezas() -> list[Path]:
    """las rutas a sellar, ordenadas y sin duplicados.

    returns:
        lista de rutas relativas a la raiz que existen en disco.
    """
    out = {RAIZ / p for p in PIEZAS if (RAIZ / p).exists()}
    for pat in PATRONES:
        out.update(RAIZ.glob(pat))
    return sorted(out)


def construye() -> dict:
    """arma el manifiesto del estado local.

    returns:
        dict con commit, fecha, y sha256 y bytes de cada pieza.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        sucio = bool(subprocess.run(
            ["git", "-C", str(RAIZ), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, sucio = "", True
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "commit": commit,
        "arbol_sucio": sucio,
        "piezas": {
            str(p.relative_to(RAIZ)): {"sha256": sha256(p),
                                       "bytes": p.stat().st_size}
            for p in piezas()},
    }


def main() -> None:
    """escribe el sello, o lo contrasta contra el disco."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true")
    args = parser.parse_args()

    if args.verificar:
        if not SALIDA.exists():
            raise SystemExit(f"no hay sello en {SALIDA}")
        viejo = json.loads(SALIDA.read_text())
        nuevo = construye()
        difieren, faltan, nuevas = [], [], []
        for k, v in viejo["piezas"].items():
            if k not in nuevo["piezas"]:
                faltan.append(k)
            elif nuevo["piezas"][k]["sha256"] != v["sha256"]:
                difieren.append(k)
        nuevas = [k for k in nuevo["piezas"] if k not in viejo["piezas"]]
        print(f"[sello] de {viejo['fecha_utc']}, commit "
              f"{viejo['commit'][:8] or 'sin commit'}")
        for etiqueta, lista in [("difieren", difieren),
                                ("faltan", faltan),
                                ("nuevas", nuevas)]:
            print(f"[sello] {etiqueta}: {len(lista)}")
            for k in lista:
                print(f"    {k}")
        if difieren or faltan:
            sys.exit(1)
        print("[sello] el disco coincide con el sello")
        return

    sello = construye()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(sello, indent=2, ensure_ascii=False))
    print(f"[sello] {len(sello['piezas'])} piezas -> {SALIDA}")
    if sello["arbol_sucio"]:
        print("[sello] AVISO: el arbol de git tiene cambios sin "
              "comitear; el sello no corresponde a ningun commit.")
    for k in ("paper/gauge_orbit_quantization_es.pdf",
              "paper/gauge_orbit_quantization_en.pdf"):
        if k in sello["piezas"]:
            print(f"[sello] {k}\n          "
                  f"{sello['piezas'][k]['sha256']}")


if __name__ == "__main__":
    main()
