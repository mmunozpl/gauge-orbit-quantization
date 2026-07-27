"""error relativo del circuito ov, recompuesto siempre en fp64.

el circuito m = w_o^(h) w_v^(h) es la cantidad gauge-invariante (ver
paper 1, sec:gauge). la nota mide cuánto se aleja la versión
cuantizada-y-recompuesta de la referencia exacta.
"""

import torch


def circuito(w_v_h: torch.Tensor, w_o_h: torch.Tensor) -> torch.Tensor:
    """circuito m = w_o^(h) w_v^(h), [d, d].

    args:
        w_v_h: tensor [d_h, d].
        w_o_h: tensor [d, d_h].

    returns:
        tensor [d, d] en la dtype de entrada.
    """
    return w_o_h @ w_v_h


def error_relativo(
    w_v_hat: torch.Tensor,
    w_o_hat: torch.Tensor,
    m_referencia: torch.Tensor,
) -> float:
    """e = ||m_hat - m_referencia||_f / ||m_referencia||_f, en fp64.

    args:
        w_v_hat: w_v^(h) tras gauge y cuantización, [d_h, d].
        w_o_hat: w_o^(h) tras gauge y cuantización, [d, d_h].
        m_referencia: circuito exacto de referencia (sin gauge, sin
            cuantizar), [d, d], en float64.

    returns:
        error relativo de frobenius, escalar float.
    """
    m_hat = circuito(w_v_hat.double(), w_o_hat.double())
    return float(
        (m_hat - m_referencia).norm() / m_referencia.norm())
