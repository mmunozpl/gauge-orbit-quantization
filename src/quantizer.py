"""cuantizador rtn simétrico por canal, con des-cuantización.

round-to-nearest puro: sin gptq, sin rotaciones aprendidas, sin
calibración con datos. la simplicidad es deliberada —aísla la
variable de interés (el gauge), la nota no compite con métodos de
cuantización, los caracteriza (ver spec, sec. 2.2).
"""

import torch

_QMAX = {4: 7, 8: 127}  # 2^(bits-1) - 1, rango simétrico con signo


def rtn_cuantiza_descuantiza(
    w: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """cuantiza y descuantiza w por canal (fila), simétrico, rtn.

    cada fila obtiene su propia escala —el "canal" es la fila, la
    convención estándar para pesos [salida, entrada]—. la aritmética
    es en la dtype de entrada (se espera float64, el suelo numérico
    del resto de la nota); no hay almacenamiento entero real, se
    simula la rejilla de cuantización.

    args:
        w: tensor [filas, columnas].
        bits: 4 u 8.

    returns:
        tensor de la misma forma y dtype, redondeado a la rejilla de
        `bits` y devuelto a escala real (des-cuantizado).
    """
    if bits not in _QMAX:
        raise ValueError(f"bits no soportado: {bits}")
    qmax = _QMAX[bits]
    escala = w.abs().amax(dim=-1, keepdim=True) / qmax
    # fila nula: escala 0/0 -> nan; se fija a 1 (la fila cuantizada es
    # 0 de todas formas, la escala no importa).
    escala = torch.where(escala > 0, escala, torch.ones_like(escala))
    q = torch.round(w / escala).clamp(min=-qmax, max=qmax)
    return q * escala
