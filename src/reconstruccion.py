"""reconstruye un gauge concreto del barrido, re-jugando la misma
secuencia de muestreo que ``scripts/sweep.py::barre_capa`` con el
mismo generador y semilla. necesario para la validación end-to-end
(sec. 2.4 del spec): identifica el mejor/peor gauge MUESTREADO por el
barrido y lo reconstruye exactamente, sin persistir cada r en el csv.
"""

import torch

from src.gauges import gl_general, ortogonal

K_MUESTRAS = 32
ESCALAS_GL = [2.0, 8.0, 32.0]


def reconstruye_r(
    capa_objetivo: int,
    cabeza_objetivo: int,
    regimen_objetivo: str,
    muestra_objetivo: int,
    n_cabezas: int,
    dim_cabeza: int,
    n_capas_previas_por_capa: dict[int, int] | None = None,
    semilla: int = 0,
    escala_objetivo: float | None = None,
) -> torch.Tensor:
    """re-juega el muestreo hasta dar con el r exacto pedido.

    reproduce el orden de ``barre_capa``: por capa, primero los k
    gauges compartidos (orto_comp), luego por cabeza los k orto_ph
    seguidos de los k·3 gl_e (por escala). se recorre capa a capa
    desde 0 hasta la objetivo, consumiendo el generador exactamente
    como el barrido original, y se descartan los r que no hacen falta.

    args:
        capa_objetivo: capa del gauge buscado.
        cabeza_objetivo: cabeza del gauge buscado (ignorada si
            regimen_objetivo es 'orto_comp', que es por-capa).
        regimen_objetivo: 'orto_ph', 'orto_comp' o 'gl_e'.
        muestra_objetivo: índice de muestra (0..31).
        n_cabezas: cabezas por capa del modelo (constante).
        dim_cabeza: d_h.
        n_capas_previas_por_capa: sin uso (todas las capas del modelo
            tienen el mismo n_cabezas); se mantiene por si algún
            modelo futuro varía cabezas por capa.
        semilla: semilla global del barrido (por defecto la del
            barrido real, ver scripts/sweep.py --semilla).
        escala_objetivo: escala de gl_e buscada (solo si
            regimen_objetivo == 'gl_e').

    returns:
        tensor [d_h, d_h] en float64, idéntico al usado en el barrido.
    """
    gen = torch.Generator().manual_seed(semilla)
    for capa in range(capa_objetivo + 1):
        compartidos = [ortogonal(dim_cabeza, gen)
                      for _ in range(K_MUESTRAS)]
        if capa == capa_objetivo and regimen_objetivo == "orto_comp":
            return compartidos[muestra_objetivo]
        ultima_capa = capa == capa_objetivo
        for h in range(n_cabezas):
            orto_ph = [ortogonal(dim_cabeza, gen)
                      for _ in range(K_MUESTRAS)]
            if (ultima_capa and h == cabeza_objetivo
                    and regimen_objetivo == "orto_ph"):
                return orto_ph[muestra_objetivo]
            for escala in ESCALAS_GL:
                gl = [gl_general(dim_cabeza, escala, gen)
                     for _ in range(K_MUESTRAS)]
                if (ultima_capa and h == cabeza_objetivo
                        and regimen_objetivo == "gl_e"
                        and escala == escala_objetivo):
                    return gl[muestra_objetivo]
    raise ValueError(
        f"no se encontró capa={capa_objetivo} cabeza={cabeza_objetivo} "
        f"regimen={regimen_objetivo} muestra={muestra_objetivo}")


def reconstruye_muchos(
    objetivos: set[tuple],
    n_capas: int,
    n_cabezas: int,
    dim_cabeza: int,
    semilla: int = 0,
) -> dict[tuple, torch.Tensor]:
    """como `reconstruye_r`, pero en una sola pasada para muchos r's.

    evita rejugar la secuencia desde cero por cada objetivo (o(n) en
    vez de o(n²)): recorre el modelo entero una vez y captura cada r
    pedido al pasar por su punto exacto.

    args:
        objetivos: conjunto de tuplas
            (capa, cabeza, regimen, muestra, escala) — escala es
            ``None`` salvo para regimen='gl_e'. cabeza se ignora para
            'orto_comp' (usar el valor que sea, ej. -1).
        n_capas: capas totales del modelo (recorre 0..n_capas-1).
        n_cabezas: cabezas por capa.
        dim_cabeza: d_h.
        semilla: semilla global del barrido.

    returns:
        dict objetivo -> tensor [d_h, d_h] en float64, con todas las
        claves de `objetivos` presentes (lanza si falta alguna).
    """
    restantes = set(objetivos)
    hallados: dict[tuple, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(semilla)
    for capa in range(n_capas):
        compartidos = [ortogonal(dim_cabeza, gen)
                      for _ in range(K_MUESTRAS)]
        for m, r in enumerate(compartidos):
            for cab in range(n_cabezas):
                clave = (capa, cab, "orto_comp", m, None)
                if clave in restantes:
                    hallados[clave] = r
        for h in range(n_cabezas):
            orto_ph = [ortogonal(dim_cabeza, gen)
                      for _ in range(K_MUESTRAS)]
            for m, r in enumerate(orto_ph):
                clave = (capa, h, "orto_ph", m, None)
                if clave in restantes:
                    hallados[clave] = r
            for escala in ESCALAS_GL:
                gl = [gl_general(dim_cabeza, escala, gen)
                     for _ in range(K_MUESTRAS)]
                for m, r in enumerate(gl):
                    clave = (capa, h, "gl_e", m, escala)
                    if clave in restantes:
                        hallados[clave] = r
    faltan = restantes - set(hallados)
    if faltan:
        raise ValueError(f"objetivos no encontrados: {faltan}")
    return hallados
