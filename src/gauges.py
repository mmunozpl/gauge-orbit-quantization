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
