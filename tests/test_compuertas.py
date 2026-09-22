"""las compuertas del proyecto, invocables desde pytest.

envoltorio fino: no duplica la lógica de `scripts/sanity.py` ni la de
`scripts/sweep_kappa.py`, las invoca.

las tres compuertas del barrido principal necesitan los pesos reales
del ViT-B, que no son redistribuibles; si la configuración local no
los declara, el test se salta con motivo explícito en vez de fallar.
la compuerta de kappa no necesita datos y corre siempre: es la que
garantiza que la familia de gauges controlada es lo que dice ser, y
de ella depende la ley del producto.

uso:
    pytest -v tests/
"""

import sys
from pathlib import Path

import pytest
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from sanity import (_modelos_muestra, sanity_gauge_sin_cuantizar,
                    sanity_idempotencia, sanity_int8_bate_int4)
from src.config import load_config
from sweep_kappa import MODELOS, TOL_KAPPA, compuerta_kappa, rejilla_kappa


@pytest.fixture(scope="module")
def modelos():
    """carga los modelos de muestra, o salta si faltan los pesos.

    returns:
        la lista de (nombre, modelo, n_cabezas, dim_cabeza) que las
        compuertas del barrido principal esperan.
    """
    cfg = load_config(str(RAIZ / "configs" / "checkpoints.yaml"))
    try:
        return _modelos_muestra(cfg)
    except ValueError as exc:
        pytest.skip(f"pesos no declarados en la configuración local: {exc}")


def test_compuerta_kappa():
    """cond(R) reproduce el kappa objetivo en toda la rejilla.

    no necesita datos: es la compuerta de la familia controlada, de la
    que depende la ley del producto, y por eso corre siempre.
    """
    gen = torch.Generator().manual_seed(0)
    compuerta_kappa(MODELOS["vitb"]["dim_cabeza"], rejilla_kappa(), gen)


def test_compuerta_kappa_rechaza_desviacion():
    """la compuerta aborta si el kappa medido no es el objetivo.

    se le pide verificar un kappa que la construcción no produce, para
    comprobar que la compuerta detecta y no solo decora.
    """
    import sweep_kappa

    gen = torch.Generator().manual_seed(0)
    original = sweep_kappa.gl_condicionado
    try:
        # se devuelve siempre un gauge de kappa 2, sea cual sea el
        # objetivo: la compuerta debe verlo
        sweep_kappa.gl_condicionado = (
            lambda dim, kappa, generador: original(dim, 2.0, generador))
        with pytest.raises(RuntimeError, match="compuerta de kappa"):
            sweep_kappa.compuerta_kappa(64, [2.0, 100.0], gen)
    finally:
        sweep_kappa.gl_condicionado = original


def test_gauge_ortogonal_sin_cuantizar(modelos):
    """gauge ortogonal sin cuantizar deja e en el suelo de fp64."""
    assert sanity_gauge_sin_cuantizar(modelos)


def test_int8_bate_int4(modelos):
    """int8 sobre la identidad da un orden de magnitud menos que int4."""
    assert sanity_int8_bate_int4(modelos)


def test_cuantizador_idempotente(modelos):
    """des-cuantizar y re-cuantizar no mueve los pesos."""
    assert sanity_idempotencia(modelos)


def test_tolerancia_declarada():
    """la tolerancia de la compuerta es la que el preregistro fijó."""
    assert TOL_KAPPA == 1e-10


def _par_sintetico(d: int = 96, dh: int = 16, semilla: int = 0):
    """par (w_v, w_o) aleatorio en fp64, sin pesos reales."""
    g = torch.Generator().manual_seed(semilla)
    w_v = torch.randn(dh, d, generator=g, dtype=torch.float64)
    w_o = torch.randn(d, dh, generator=g, dtype=torch.float64)
    return w_v, w_o


def test_clase_conforme_es_neutra():
    """p(c·Q) = 1 a precisión de máquina, y d(c·Q) = 0.

    la clase inocua del paper 1 y la clase neutra de la ley del
    producto son la misma; no necesita datos y corre siempre.
    """
    from punto_balanceado import compuerta_conforme

    gen = torch.Generator().manual_seed(0)
    compuerta_conforme(64, gen)


def test_gauge_balanceado_preserva_el_circuito():
    """el punto balanceado es un gauge: el circuito no se mueve."""
    from src.gauges import aplica_gauge, gauge_balanceado
    from src.metrics import circuito

    w_v, w_o = _par_sintetico()
    m = circuito(w_v, w_o)
    r, _ = gauge_balanceado(w_v, w_o)
    w_v_b, w_o_b = aplica_gauge(w_v, w_o, r)
    deriva = float((circuito(w_v_b, w_o_b) - m).norm() / m.norm())
    assert deriva < 1e-12, f"el circuito se movió: {deriva:.2e}"
    # y el par queda balanceado: A^T A = B B^T
    gram_a = w_o_b.transpose(-2, -1) @ w_o_b
    gram_b = w_v_b @ w_v_b.transpose(-2, -1)
    desbalance = float((gram_a - gram_b).norm() / gram_a.norm())
    assert desbalance < 1e-10, f"no está balanceado: {desbalance:.2e}"


def test_punto_balanceado_minimiza_el_producto():
    """ningún gauge muestreado baja de p del punto balanceado.

    es la desigualdad ||m||_* <= ||a||_f ||b||_f comprobada: el punto
    balanceado alcanza la igualdad y por eso es el mínimo de la ley
    del producto sobre la órbita.
    """
    from src.gauges import gauge_balanceado, ortogonal, producto_normas

    w_v, w_o = _par_sintetico()
    _, p_bal = gauge_balanceado(w_v, w_o)
    g = torch.Generator().manual_seed(1)
    dh = w_v.shape[0]
    for _ in range(100):
        escalas = torch.rand(dh, generator=g, dtype=torch.float64) + 0.5
        r = ortogonal(dh, g) @ torch.diag(escalas)
        assert producto_normas(w_v, w_o, r) >= p_bal - 1e-12


def test_gauge_balanceado_rechaza_rango_deficiente():
    """sin rango completo el punto balanceado no es alcanzable."""
    from src.gauges import gauge_balanceado

    w_v, w_o = _par_sintetico()
    w_v[-1] = 0.0  # circuito de rango d_h - 1
    with pytest.raises(ValueError, match="rango deficiente"):
        gauge_balanceado(w_v, w_o)


def test_sesgo_de_valor_viaja_con_el_gauge(modelos):
    """sin transformar b_v, un gauge ortogonal cambia la función.

    prueba de regresión del fallo corregido el 22-09-2026: la
    validación end-to-end aplicaba el gauge al par y dejaba el sesgo
    de valor quieto, y el forward pasaba a medir otro modelo. se exige
    que la invariancia falle sin el sesgo y se cumpla con él.
    """
    import copy

    from src.config import load_config
    from src.gauges import ortogonal
    from src.weights import (b_v_vitb, cargar_vitb, escribe_b_v_vitb,
                             escribe_v_o_vitb, w_v_w_o_vitb)

    cfg = load_config(str(RAIZ / "configs" / "checkpoints.yaml"))
    base = cargar_vitb(cfg["vitb_ckpt"])
    if base.blocks[0].attn.qkv.bias is None:
        pytest.skip("la columna no lleva qkv_bias")
    n_capas, n_cabezas, dim_cabeza = 12, 12, 64

    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        y0 = base(x)

    g = torch.Generator().manual_seed(1)
    rs, sin_sesgo = {}, copy.deepcopy(base)
    for capa in range(n_capas):
        w_v, w_o = w_v_w_o_vitb(sin_sesgo, capa, n_cabezas, dim_cabeza)
        for h in range(n_cabezas):
            r = ortogonal(dim_cabeza, g)
            rs[(capa, h)] = r
            escribe_v_o_vitb(
                sin_sesgo, capa, h, r.transpose(-2, -1) @ w_v[h],
                w_o[h] @ torch.linalg.inv(r).transpose(-2, -1), dim_cabeza)
    con_sesgo = copy.deepcopy(sin_sesgo)
    for capa in range(n_capas):
        b_v = b_v_vitb(base, capa, n_cabezas, dim_cabeza)
        for h in range(n_cabezas):
            escribe_b_v_vitb(
                con_sesgo, capa, h,
                rs[(capa, h)].transpose(-2, -1) @ b_v[h], dim_cabeza)

    with torch.no_grad():
        desv_sin = float((sin_sesgo(x) - y0).norm() / y0.norm())
        desv_con = float((con_sesgo(x) - y0).norm() / y0.norm())
    assert desv_sin > 1e-2, (
        f"el fallo del sesgo ya no se detecta: {desv_sin:.2e}")
    assert desv_con < 1e-4, (
        f"con el sesgo transformado la invariancia falla: {desv_con:.2e}")


def test_los_minimos_son_la_familia_balanceada_salvo_escalado():
    """el conjunto de mínimos de p, tal como el corolario lo enuncia.

    tres comprobaciones. el escalado recíproco deja p invariante y
    rompe el balance, así que el mínimo NO es solo el punto
    balanceado. la parte ortogonal tampoco lo mueve. y el recíproco
    de la demostración: reescalar cualquier mínimo a normas iguales
    devuelve el balance, que es por qué no hay más mínimos.
    """
    from src.gauges import (aplica_gauge, gauge_balanceado, ortogonal,
                            producto_normas)

    w_v, w_o = _par_sintetico()
    r_bal, p_bal = gauge_balanceado(w_v, w_o)
    g = torch.Generator().manual_seed(2)

    for c in (0.01, 0.5, 3.0, 250.0):
        p_c = producto_normas(w_v, w_o, c * r_bal)
        assert abs(p_c - p_bal) < 1e-12, f"c={c} movió p: {p_c - p_bal:.2e}"
        a, b = aplica_gauge(w_v, w_o, c * r_bal)
        gram_a, gram_b = b.transpose(-2, -1) @ b, a @ a.transpose(-2, -1)
        desbalance = float((gram_a - gram_b).norm() / gram_a.norm())
        if c != 1.0:
            assert desbalance > 1e-3, (
                f"c={c} debería romper el balance: {desbalance:.2e}")

    for _ in range(5):
        p_q = producto_normas(w_v, w_o, r_bal @ ortogonal(w_v.shape[0], g))
        assert abs(p_q - p_bal) < 1e-12

    # recíproco: normas iguales => balanceado
    for c in (0.2, 7.0):
        a, b = aplica_gauge(w_v, w_o, c * r_bal)
        c_est = (float(b.norm()) / float(a.norm())) ** 0.5
        a2, b2 = c_est * a, b / c_est
        assert abs(float(a2.norm()) - float(b2.norm())) < 1e-9
        gram_a, gram_b = b2.transpose(-2, -1) @ b2, a2 @ a2.transpose(-2, -1)
        assert float((gram_a - gram_b).norm() / gram_a.norm()) < 1e-10


def test_lema_del_cuantizador():
    """la cota del lema 1 se cumple: ||dW||_F <= sqrt(n)/(2qmax) ||W||_F.

    el lema alimenta la hipótesis de la proposición, así que conviene
    que sea comprobable y no solo demostrable. se prueban escalas de
    1e-6 a 1e6 y filas nulas, que son los dos casos donde una cota de
    este tipo suele romperse.
    """
    from src.quantizer import rtn_cuantiza_descuantiza

    g = torch.Generator().manual_seed(0)
    peor = 0.0
    for _ in range(200):
        m = int(torch.randint(1, 40, (1,), generator=g))
        n = int(torch.randint(1, 60, (1,), generator=g))
        escala = 10.0 ** int(torch.randint(-6, 6, (1,), generator=g))
        w = torch.randn(m, n, generator=g, dtype=torch.float64) * escala
        if float(torch.rand(1, generator=g)) < 0.2:
            w[0] = 0.0          # fila nula, el caso degenerado
        for bits, qmax in ((4, 7), (8, 127)):
            dif = float((w - rtn_cuantiza_descuantiza(w, bits)).norm())
            cota = (n ** 0.5) / (2 * qmax) * float(w.norm())
            assert dif <= cota + 1e-12, (
                f"la cota del lema falla: {dif:.3e} > {cota:.3e}")
            if cota > 0:
                peor = max(peor, dif / cota)
    assert peor > 0.05, "la cota es tan holgada que no prueba nada"
