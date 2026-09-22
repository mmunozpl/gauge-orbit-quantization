"""construcción de gauges valor-salida en fp64.

reutiliza la construcción del paper 1 (angular-separation-vit):
ortogonales vía qr de una gaussiana (haar), gl generales vía
randn + escala·identidad. toda la aritmética en float64 —la lección
del r casi-singular del paper 1 es constitutiva aquí, no una nota al
pie.
"""

import torch


def ortogonal(
    dim_cabeza: int,
    generador: torch.Generator,
) -> torch.Tensor:
    """muestra una rotación haar-aleatoria en o(dim_cabeza), fp64.

    args:
        dim_cabeza: d_h, tamaño de la matriz cuadrada.
        generador: generador de torch para reproducibilidad.

    returns:
        tensor [d_h, d_h] ortogonal en float64.
    """
    a = torch.randn(dim_cabeza, dim_cabeza, generator=generador,
                    dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # signo de la diagonal de r fija el signo de q (convención qr no
    # es única si no se corrige); sin esto la distribución no es haar.
    signos = torch.sign(torch.diagonal(r))
    signos[signos == 0] = 1.0
    return q * signos


def gl_general(
    dim_cabeza: int,
    escala: float,
    generador: torch.Generator,
) -> torch.Tensor:
    """muestra r = randn + escala·i en gl(dim_cabeza), fp64.

    escala alta acerca r a un múltiplo escalar de la identidad (gauge
    débil, bien condicionado); escala baja lo aleja (gauge fuerte,
    condición alta, la cola que la nota espera que cuantice mal).

    args:
        dim_cabeza: d_h.
        escala: coeficiente de la identidad.
        generador: generador de torch para reproducibilidad.

    returns:
        tensor [d_h, d_h] en float64, invertible con probabilidad 1.
    """
    ident = torch.eye(dim_cabeza, dtype=torch.float64)
    ruido = torch.randn(dim_cabeza, dim_cabeza, generator=generador,
                        dtype=torch.float64)
    return ruido + escala * ident


def aplica_gauge(
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """transforma (w_v^(h), w_o^(h)) bajo el gauge r, sin tocar el circuito.

    convención: w_v^(h) es [d_h, d] (filas de valor), w_o^(h) es
    [d, d_h] (columnas de salida). w_v <- r^t w_v, w_o <- w_o r^-t;
    el circuito m = w_o w_v es invariante para cualquier r en
    gl(d_h) (no solo ortogonal): m' = (w_o r^-t)(r^t w_v)
    = w_o (r^-t r^t) w_v = w_o w_v, ya que r^-t r^t = (r^t)^-1 r^t = i.

    args:
        w_v_h: tensor [d_h, d] en float64.
        w_o_h: tensor [d, d_h] en float64.
        r: gauge [d_h, d_h] en float64, invertible.

    returns:
        (w_v_h transformado, w_o_h transformado), misma forma.
    """
    r_inv_t = torch.linalg.inv(r).transpose(-2, -1)
    return r.transpose(-2, -1) @ w_v_h, w_o_h @ r_inv_t


def gl_condicionado(
    dim_cabeza: int,
    kappa: float,
    generador: torch.Generator,
) -> torch.Tensor:
    """muestra r en gl(dim_cabeza) con número de condición kappa exacto.

    se toma una gaussiana, se le extrae la svd y se sustituye su
    espectro por uno geométrico entre 1 y kappa: las bases quedan
    haar-aleatorias y el condicionamiento pasa a ser la variable
    controlada, en vez de una consecuencia de la escala. la
    alternativa declarada y no elegida es el espectro de un solo
    outlier (sigma_1 = kappa, resto 1).

    kappa = 1 devuelve una matriz ortogonal, que es el ancla del
    suelo e_0 del ajuste.

    args:
        dim_cabeza: d_h.
        kappa: número de condición objetivo, >= 1.
        generador: generador de torch para reproducibilidad.

    returns:
        tensor [d_h, d_h] en float64 con cond(r) = kappa.

    raises:
        ValueError: si kappa < 1.
    """
    if kappa < 1.0:
        raise ValueError(f"kappa debe ser >= 1: {kappa}")
    a = torch.randn(dim_cabeza, dim_cabeza, generator=generador,
                    dtype=torch.float64)
    u, _, vh = torch.linalg.svd(a)
    # espectro geométrico descendente de kappa a 1; el cociente entre
    # el mayor y el menor es kappa por construcción.
    exponentes = torch.linspace(1.0, 0.0, dim_cabeza, dtype=torch.float64)
    s = torch.pow(torch.tensor(kappa, dtype=torch.float64), exponentes)
    return (u * s) @ vh


def producto_normas(
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    r: torch.Tensor | None = None,
) -> float:
    """p(r) = ||r^t w_v||_f ||w_o r^-t||_f / (||w_v||_f ||w_o||_f).

    la variable de la ley del producto: el inflado del error de
    cuantización a primer orden. vale 1 en la identidad y 1 en toda
    la clase conforme ortogonal {c·q}, porque ||w_v (cq)|| escala por
    c y ||(cq)^-1 w_o|| por 1/c.

    args:
        w_v_h: tensor [d_h, d] en float64, sin gauge.
        w_o_h: tensor [d, d_h] en float64, sin gauge.
        r: gauge [d_h, d_h]; None denota la identidad (p = 1).

    returns:
        p(r), escalar float.
    """
    base = float(w_v_h.norm() * w_o_h.norm())
    if r is None:
        return 1.0
    w_v_g, w_o_g = aplica_gauge(w_v_h, w_o_h, r)
    return float(w_v_g.norm() * w_o_g.norm()) / base


def gauge_balanceado(
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    tol_rango: float = 1e-10,
) -> tuple[torch.Tensor, float]:
    """el gauge que lleva el par a su factorización balanceada.

    el punto balanceado del paper 1 (lema del punto balanceado)
    minimiza ||w_v r||_f^2 + ||r^-1 w_o||_f^2 sobre la órbita a
    circuito fijo, y se construye por svd del circuito: con
    m = w_o w_v = u s v^t de rango d_h, el par balanceado es
    a = u s^{1/2} y b = s^{1/2} v^t.

    el enlace con esta nota es exacto: ||a||_f ||b||_f = tr(s) =
    ||m||_*, que es el mínimo del producto de normas sobre la órbita
    entera (desigualdad de la norma nuclear, ||m||_* <= ||a||_f
    ||b||_f para toda factorización m = ab). el punto balanceado es
    entonces el óptimo que la ley del producto predice, y es un gauge
    NO ortogonal en general: el contraejemplo medido a «la
    restricción ortogonal es necesaria».

    convención de este repo: m = w_o w_v, y el gauge actúa como
    w_v <- r^t w_v, w_o <- w_o r^-t. con s = r^t se pide
    w_o s^-1 = a y s w_v = b.

    args:
        w_v_h: tensor [d_h, d] en float64.
        w_o_h: tensor [d, d_h] en float64.
        tol_rango: valor singular mínimo relativo para declarar el
            circuito de rango completo.

    returns:
        (r [d_h, d_h] en float64, p_balanceado), donde p_balanceado
        es ||m||_* / (||w_v||_f ||w_o||_f) <= 1, el valor de la ley
        del producto en ese punto.

    raises:
        ValueError: si el circuito no tiene rango d_h, donde el punto
            balanceado no es alcanzable por un gauge de gl(d_h).
    """
    d_h = w_v_h.shape[0]
    m = w_o_h @ w_v_h
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    u, s, vh = u[:, :d_h], s[:d_h], vh[:d_h]
    if float(s[-1]) <= tol_rango * float(s[0]):
        raise ValueError(
            f"circuito de rango deficiente: s_min/s_max = "
            f"{float(s[-1]) / float(s[0]):.2e} <= {tol_rango:.0e}")
    raiz = s.sqrt()
    a_bal = u * raiz            # [d, d_h]
    # s^-1 = w_o^+ a_bal; r = s^t = (s^-1)^-t
    s_inv = torch.linalg.pinv(w_o_h) @ a_bal
    r = torch.linalg.inv(s_inv).transpose(-2, -1)
    p_bal = float(s.sum()) / float(w_v_h.norm() * w_o_h.norm())
    return r, p_bal
