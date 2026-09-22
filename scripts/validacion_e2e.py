"""validación end-to-end (sec. 2.4 del spec, regla actualizada 27-07).

el prereg original pedía coherencia de signo entre identidad/mejor/
peor gauge ortogonal muestreado. q1 mostró que esos tres puntos
difieren en e apenas un 2-8 %: a esa distancia tres forwards no
pueden resolver un signo, y encontrar "incoherencia" sería ruido
leído como señal. regla actualizada, fijada antes de correr: si el
rango ortogonal es <×1,25 (lo es), la predicción pasa a ser
INDISTINGUIBILIDAD —los tres top-1 dentro del ruido—, y eso confirma
q1 end-to-end.

corrección del 22-09-2026. el sesgo de valor NO se transformaba con
el par: como v = x w_v^t + b_v, la invariancia exige b_v <- r^t b_v,
y omitirlo convierte el forward en el de otro modelo. medido sobre un
lote sin cuantizar, un gauge ortogonal movía la salida un 57 %
relativo y dejaba coincidir el 12 % de los top-1; con el sesgo
transformado la desviación baja a 5e-7. la tabla anterior atribuía a
la identidad una ventaja de 2,8 puntos que era ese sesgo. ahora el
sesgo viaja con el gauge y una compuerta de invariancia lo verifica
ANTES de cuantizar, condición a condición.

cuatro condiciones nuevas respecto a la tabla anterior: los dos
controles sin cuantizar —identidad y gl de escala 2— que aíslan la
cuantización del gauge, y el punto balanceado, el gauge NO ortogonal
que la ley del producto predice como óptimo de la órbita.

los gauges muestreados se reconstruyen exactos vía replay del
generador (src/reconstruccion), no se re-muestrean: son los mismos r
que el barrido ya midió. el balanceado se construye por svd del
circuito (src/gauges.gauge_balanceado).

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
from src.datos import predicciones_y_aciertos, val_loader_imagenet100
from src.gauges import gauge_balanceado
from src.quantizer import rtn_cuantiza_descuantiza
from src.reconstruccion import reconstruye_muchos
from src.weights import (b_v_vitb, cargar_vitb, escribe_b_v_vitb,
                         escribe_v_o_vitb, w_v_w_o_vitb)

N_CAPAS, N_CABEZAS, DIM_CABEZA = 12, 12, 64
BITS = 4
BARRIDO_CSV = "artifacts/logs/quant_orbita/quant_orbita.csv"
SALIDA = "artifacts/tables/validacion_e2e.csv"
SALIDA_ACIERTOS = "artifacts/logs/quant_orbita/aciertos_e2e.pt"
SALIDA_PRED = "artifacts/logs/quant_orbita/predicciones_e2e.pt"
GL_ESCALA_TOXICA = 2.0
GL_MUESTRA_REPRESENTATIVA = 0  # fijo, no elegido por ser el peor
# compuerta de invariancia: con el sesgo transformado, un gauge bien
# condicionado deja la salida a ~5e-7 en fp32. el listón se fija dos
# órdenes por encima, y solo es exigible donde el gauge está bien
# condicionado; en gl de escala 2 la desviación se mide y se reporta,
# porque esa degradación es ella misma un resultado.
TOL_INVARIANCIA = 1e-4
EXIGEN_INVARIANCIA = ("mejor_orto_muestreado", "peor_orto_muestreado",
                      "balanceado")


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


def gauges_balanceados(cfg: dict) -> tuple[dict, float]:
    """el gauge del punto balanceado de cada cabeza, y su p mediano.

    args:
        cfg: config con la ruta del checkpoint.

    returns:
        (dict (capa, cabeza) -> r [d_h, d_h] float64, mediana de p).
    """
    modelo = cargar_vitb(cfg["vitb_ckpt"])
    gauges, ps = {}, []
    for capa in tqdm(range(N_CAPAS), desc="punto balanceado"):
        w_v, w_o = w_v_w_o_vitb(modelo, capa, N_CABEZAS, DIM_CABEZA)
        for h in range(N_CABEZAS):
            r, p_bal = gauge_balanceado(w_v[h], w_o[h])
            gauges[(capa, h)] = r
            ps.append(p_bal)
    del modelo
    return gauges, float(pd.Series(ps).median())


@torch.no_grad()
def _construye_modelo(
    cfg: dict,
    disp: str,
    gauges: dict[tuple, torch.Tensor] | None,
    bits: int | None = BITS,
) -> torch.nn.Module:
    """carga vit-b fresco y aplica gauge (+cuantización) por cabeza.

    el sesgo de valor se transforma junto al par: sin eso el gauge
    deja de ser un cambio de coordenadas. el sesgo no se cuantiza —el
    objeto cuantizado de la nota es el par de weights—.

    args:
        cfg: config con la ruta del checkpoint.
        disp: dispositivo destino.
        gauges: dict (capa, cabeza) -> r [d_h, d_h] float64, o None
            para la condición identidad (r = i implícita).
        bits: anchura de cuantización, o None para no cuantizar.

    returns:
        el modelo, en eval, sobre disp, con los pesos ya modificados.
    """
    modelo = cargar_vitb(cfg["vitb_ckpt"]).to(disp)
    for capa in range(N_CAPAS):
        w_v, w_o = w_v_w_o_vitb(modelo, capa, N_CABEZAS, DIM_CABEZA)
        b_v = b_v_vitb(modelo, capa, N_CABEZAS, DIM_CABEZA)
        for h in range(N_CABEZAS):
            w_v_h, w_o_h = w_v[h], w_o[h]
            if gauges is not None:
                r = gauges[(capa, h)].to(w_v_h.device)
                r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
                w_v_h = r.transpose(-2, -1) @ w_v_h
                w_o_h = w_o_h @ r_inv_t
                if b_v is not None:
                    escribe_b_v_vitb(
                        modelo, capa, h,
                        r.transpose(-2, -1) @ b_v[h].to(r.device),
                        DIM_CABEZA)
            if bits is not None:
                w_v_h = rtn_cuantiza_descuantiza(w_v_h, bits)
                w_o_h = rtn_cuantiza_descuantiza(w_o_h, bits)
            escribe_v_o_vitb(modelo, capa, h, w_v_h, w_o_h, DIM_CABEZA)
    return modelo.eval()


@torch.no_grad()
def compuerta_invariancia(
    cfg: dict,
    disp: str,
    gauges: dict,
    lote: torch.Tensor,
    logits_ref: torch.Tensor,
) -> float:
    """desviación relativa del forward bajo el gauge, SIN cuantizar.

    la invariancia es exacta en aritmética real: si el forward se
    mueve, el gauge no se aplicó entero (el caso del sesgo) o la
    precisión finita ya lo está rompiendo antes de cuantizar.

    args:
        cfg: config con la ruta del checkpoint.
        disp: dispositivo.
        gauges: dict (capa, cabeza) -> r.
        lote: lote fijo de imágenes.
        logits_ref: logits del modelo sin gauge ni cuantizar.

    returns:
        ||logits_gauge - logits_ref||_f / ||logits_ref||_f.
    """
    modelo = _construye_modelo(cfg, disp, gauges, bits=None)
    logits = modelo(lote.to(disp))
    del modelo
    return float((logits - logits_ref).norm() / logits_ref.norm())


def main() -> None:
    """corre los siete forwards y guarda la tabla y los aciertos."""
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
    gauges_bal, p_bal_mediano = gauges_balanceados(cfg)
    print(f"[e2e] punto balanceado: p mediano = {p_bal_mediano:.4f} "
         f"(la ley predice e/e_ident ≈ p)", flush=True)

    print("[e2e] construyendo loader de val...", flush=True)
    loader = val_loader_imagenet100(
        cfg["imagenet100_root"], cfg["imagenet100_wnids"])

    # lote fijo y logits de referencia para la compuerta de invariancia
    lote = next(iter(loader))[0]
    modelo_ref = _construye_modelo(cfg, disp, None, bits=None)
    logits_ref = modelo_ref(lote.to(disp))
    del modelo_ref

    condiciones = [
        ("identidad_fp32", None, None, False),
        ("identidad", None, BITS, False),
        ("mejor_orto_muestreado", gauges_mejor, BITS, False),
        ("peor_orto_muestreado", gauges_peor, BITS, False),
        ("balanceado", gauges_bal, BITS, True),
        ("gl_escala2_fp32", gauges_gl2, None, True),
        ("gl_escala2", gauges_gl2, BITS, True),
    ]
    filas, aciertos, predichas = [], {}, {}
    for nombre, gauges, bits, post_hoc in condiciones:
        desv = ""
        if gauges is not None:
            desv = compuerta_invariancia(cfg, disp, gauges, lote,
                                        logits_ref)
            print(f"[e2e] {nombre}: invariancia pre-cuantización = "
                 f"{desv:.2e}", flush=True)
            if (nombre in EXIGEN_INVARIANCIA
                    and desv >= TOL_INVARIANCIA):
                raise RuntimeError(
                    f"compuerta de invariancia fallida en '{nombre}': "
                    f"{desv:.2e} >= {TOL_INVARIANCIA:.0e}; el gauge no "
                    f"se está aplicando entero")
        print(f"[e2e] condición '{nombre}'...", flush=True)
        modelo = _construye_modelo(cfg, disp, gauges, bits)
        pred, ok = predicciones_y_aciertos(
            modelo, tqdm(loader, desc=nombre, leave=False), disp)
        aciertos[nombre] = ok
        predichas[nombre] = pred
        acc = float(ok.double().mean())
        filas.append({"condicion": nombre,
                      "bits": "" if bits is None else bits,
                      "top1": acc, "desv_invariancia": desv,
                      "post_hoc_declarado": post_hoc})
        print(f"[e2e]   top1 = {acc:.4f}", flush=True)
        del modelo
        if disp == "cuda":
            torch.cuda.empty_cache()

    out = Path(SALIDA)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    out_ok = Path(SALIDA_ACIERTOS)
    out_ok.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aciertos, out_ok)
    out_pred = Path(SALIDA_PRED)
    torch.save(predichas, out_pred)

    print(f"\n[e2e] tabla guardada en {out}")
    print(f"[e2e] aciertos por imagen en {out_ok} y clases "
         f"predichas en {out_pred} "
         f"(contraste pareado: scripts/contraste_e2e.py)")
    for f in filas:
        print(f)

    por = {f["condicion"]: f["top1"] for f in filas}
    orto = [por["mejor_orto_muestreado"], por["peor_orto_muestreado"]]
    rango = max(orto + [por["identidad"]]) - min(orto + [por["identidad"]])
    print(f"\n[e2e] rango top1 entre identidad/mejor/peor ortogonal: "
         f"{rango:.4f} (indistinguibilidad esperada tras q1)")
    print(f"[e2e] balanceado - identidad: "
         f"{por['balanceado'] - por['identidad']:+.4f} "
         f"(p mediano {p_bal_mediano:.4f})")
    print(f"[e2e] gl escala 2 sin cuantizar: {por['gl_escala2_fp32']:.4f} "
         f"(control: aísla la cuantización del gauge)")
    igual = float((predichas["identidad_fp32"]
                   == predichas["gl_escala2_fp32"]).double().mean())
    print(f"[e2e] acuerdo de clase predicha identidad/gl sin cuantizar: "
         f"{igual:.4f} ({int(igual * len(predichas['identidad_fp32']))}"
         f"/{len(predichas['identidad_fp32'])})")
    print(f"[e2e] caída gl_escala2 vs identidad: "
         f"{por['identidad'] - por['gl_escala2']:.4f}")


if __name__ == "__main__":
    main()
