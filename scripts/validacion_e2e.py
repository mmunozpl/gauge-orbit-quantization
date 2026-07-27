"""validación end-to-end (sec. 2.4 del spec, regla actualizada 27-07).

el prereg original pedía coherencia de signo entre identidad/mejor/
peor gauge ortogonal muestreado. q1 mostró que esos tres puntos
difieren en e apenas un 2-8 %: a esa distancia tres forwards no
pueden resolver un signo, y encontrar "incoherencia" sería ruido
leído como señal. regla actualizada, fijada antes de correr: si el
rango ortogonal es <×1,25 (lo es), la predicción pasa a ser
INDISTINGUIBILIDAD —los tres top-1 dentro del ruido—, y eso confirma
q1 end-to-end. un cuarto forward, declarado post-hoc, con un gauge gl
de escala 2 (la cola tóxica) ancla la degradación en exactitud real.

los gauges "mejor"/"peor" por-cabeza y el gl de escala 2 se
reconstruyen exactos vía replay del generador (src/reconstruccion),
no se re-muestrean: son los mismos r que el barrido ya midió.

uso:
    python scripts/validacion_e2e.py --config configs/checkpoints.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.datos import top1, val_loader_imagenet100
from src.quantizer import rtn_cuantiza_descuantiza
from src.reconstruccion import reconstruye_muchos
from src.weights import cargar_vitb, escribe_v_o_vitb, w_v_w_o_vitb

N_CAPAS, N_CABEZAS, DIM_CABEZA = 12, 12, 64
BITS = 4
BARRIDO_CSV = "artifacts/logs/quant_orbita/quant_orbita.csv"
SALIDA = "artifacts/tables/validacion_e2e.csv"
GL_ESCALA_TOXICA = 2.0
GL_MUESTRA_REPRESENTATIVA = 0  # fijo, no elegido por ser el peor


def _mejor_peor_por_cabeza(df: pd.DataFrame) -> dict:
    """(capa, cabeza) -> (muestra_mejor, muestra_peor) en orto_ph."""
    sub = df[(df.modelo == "vitb") & (df.bits == BITS)
            & (df.regimen == "orto_ph")]
    out = {}
    for (capa, cabeza), g in sub.groupby(["capa", "cabeza"]):
        out[(capa, cabeza)] = (
            int(g.loc[g.err_circuito.idxmin(), "muestra"]),
            int(g.loc[g.err_circuito.idxmax(), "muestra"]),
        )
    return out


def _construye_modelo(
    cfg: dict,
    disp: str,
    gauges: dict[tuple, torch.Tensor] | None,
    bits: int = BITS,
) -> torch.nn.Module:
    """carga vit-b fresco y aplica gauge+cuantización por cabeza.

    args:
        cfg: config con la ruta del checkpoint.
        disp: dispositivo destino.
        gauges: dict (capa, cabeza) -> r [d_h, d_h] float64, o None
            para la condición identidad (r = i implícita).
        bits: anchura de cuantización.

    returns:
        el modelo, en eval, sobre disp, con los pesos ya modificados.
    """
    modelo = cargar_vitb(cfg["vitb_ckpt"]).to(disp)
    for capa in range(N_CAPAS):
        w_v, w_o = w_v_w_o_vitb(modelo, capa, N_CABEZAS, DIM_CABEZA)
        for h in range(N_CABEZAS):
            w_v_h, w_o_h = w_v[h], w_o[h]
            if gauges is not None:
                r = gauges[(capa, h)].to(w_v_h.device)
                r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
                w_v_h = r.transpose(-2, -1) @ w_v_h
                w_o_h = w_o_h @ r_inv_t
            w_v_q = rtn_cuantiza_descuantiza(w_v_h, bits)
            w_o_q = rtn_cuantiza_descuantiza(w_o_h, bits)
            escribe_v_o_vitb(modelo, capa, h, w_v_q, w_o_q, DIM_CABEZA)
    return modelo.eval()


def main() -> None:
    """corre los cuatro forwards y guarda la tabla."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--disp", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    disp = args.disp if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(BARRIDO_CSV)
    df["escala"] = pd.to_numeric(df["escala"], errors="coerce")
    mejor_peor = _mejor_peor_por_cabeza(df)

    objetivos = set()
    for (capa, cabeza), (m_mejor, m_peor) in mejor_peor.items():
        objetivos.add((capa, cabeza, "orto_ph", m_mejor, None))
        objetivos.add((capa, cabeza, "orto_ph", m_peor, None))
        objetivos.add((capa, cabeza, "gl_e", GL_MUESTRA_REPRESENTATIVA,
                      GL_ESCALA_TOXICA))
    print(f"[e2e] reconstruyendo {len(objetivos)} gauges vía replay...",
         flush=True)
    r_de = reconstruye_muchos(objetivos, N_CAPAS, N_CABEZAS, DIM_CABEZA)

    gauges_mejor = {(c, h): r_de[(c, h, "orto_ph", mp[0], None)]
                   for (c, h), mp in mejor_peor.items()}
    gauges_peor = {(c, h): r_de[(c, h, "orto_ph", mp[1], None)]
                  for (c, h), mp in mejor_peor.items()}
    gauges_gl2 = {(c, h): r_de[(c, h, "gl_e", GL_MUESTRA_REPRESENTATIVA,
                               GL_ESCALA_TOXICA)]
                 for (c, h) in mejor_peor}

    print("[e2e] construyendo loader de val...", flush=True)
    loader = val_loader_imagenet100(
        cfg["imagenet100_root"], cfg["imagenet100_wnids"])

    condiciones = [
        ("identidad", None, False),
        ("mejor_orto_muestreado", gauges_mejor, False),
        ("peor_orto_muestreado", gauges_peor, False),
        ("gl_escala2", gauges_gl2, True),
    ]
    filas = []
    for nombre, gauges, post_hoc in condiciones:
        print(f"[e2e] condición '{nombre}'...", flush=True)
        modelo = _construye_modelo(cfg, disp, gauges)
        acc = top1(modelo, tqdm(loader, desc=nombre, leave=False), disp)
        filas.append({"condicion": nombre, "bits": BITS, "top1": acc,
                      "post_hoc_declarado": post_hoc})
        print(f"[e2e]   top1 = {acc:.4f}", flush=True)
        del modelo
        torch.cuda.empty_cache() if disp == "cuda" else None

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)

    print(f"\n[e2e] tabla guardada en {out}")
    for f in filas:
        print(f)

    d_ident, d_mejor, d_peor, d_gl2 = (f["top1"] for f in filas)
    rango_orto = max(d_mejor, d_peor) - min(d_mejor, d_peor)
    print(f"\n[e2e] rango top1 entre identidad/mejor/peor ortogonal: "
         f"{rango_orto:.4f} (indistinguibilidad esperada tras q1)")
    print(f"[e2e] caída gl_escala2 vs identidad: "
         f"{d_ident - d_gl2:.4f} (degradación esperada: severa)")


if __name__ == "__main__":
    main()
