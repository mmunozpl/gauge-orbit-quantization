# [Preprint] Plana por dentro, tóxica por fuera: la órbita de gauge valor-salida bajo cuantización

🇪🇸 Español · 🇬🇧 [English](README.md)

**Manuel Muñoz Plá** · [ORCID 0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5714--912X-a6ce39)](https://orcid.org/0009-0000-5714-912X)
[![License](https://img.shields.io/badge/License-Apache--2.0-009e73)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#cómo-citar)

**Resumen:** Los métodos de cuantización rotacional explotan una invariancia
computacional del sector valor-salida de la atención: rotar el par
(*W_v*, *W_O*) sin alterar la función. Esa invariancia es un subgrupo de una
órbita mayor —*GL(d_h)* por cabeza, certificada en el trabajo previo— y la
práctica la restringe dos veces: a rotaciones ortogonales, y compartidas
entre las cabezas de la capa. Esta nota mide qué compran ambas restricciones
sobre el error de cuantización del circuito OV en dos arquitecturas
(Pythia-410M y un ViT-B/16 afinado), con un cuantizador deliberadamente
simple que aísla la variable de interés. Tres resultados. Dentro del subgrupo
ortogonal la órbita es plana: el rango de error entre rotaciones queda en
×1,01–1,02 en mediana, y el gauge entrenado ya está a ×1,02–1,04 del mejor
punto muestreado. La libertad por-cabeza no mejora la rotación compartida
(mediana de mejora ≈ 0 %). Y fuera del subgrupo, la órbita general degrada el
error de forma monótona con la fuerza del gauge, hasta ×15 en mediana y ×10⁵
en el extremo. La conclusión es asimétrica: la restricción ortogonal de la
práctica es necesaria, no una conveniencia; el punto elegido dentro de ella
es casi indiferente para los pesos, a estas escalas. Una validación
end-to-end confirma la llanura ortogonal en exactitud real —mejor y peor
rotación muestreada dan un top-1 indistinguible entre sí, con un rango de
0,08 puntos porcentuales—, aunque revela también, como observación no
preregistrada, que la identidad conserva una ventaja pequeña y consistente
sobre cualquier gauge ortogonal (≈ 2,8 puntos), señal de que el error del
circuito no agota toda la historia. Más contundente aún: un gauge fuera del
subgrupo, con el circuito exacto matemáticamente intacto en fp64, colapsa la
exactitud al nivel del azar (1,00 % sobre 100 clases) —la cuantización
convierte una elección de coordenadas en un clasificador aleatorio—. El
resultado acota además dónde no está la ganancia de los métodos rotacionales
—no en el paisaje de pesos de la órbita—, coherente con su atribución
habitual a los outliers de activación.

**Palabras clave:** cuantización, invariancia computacional, órbita de gauge,
circuito OV, atención multi-cabeza, RTN.

## Contenido

```
.
├── src/          # gauges en fp64, cuantizador RTN, error del circuito,
│                 # pesos por cabeza y reconstrucción por replay
├── scripts/      # sanities, barrido, lectura en frío, validación
│                 # end-to-end y figuras
├── configs/      # rutas a modelos y datos, nunca en el código
└── artifacts/    # tablas de lectura y figuras
```

El csv crudo del barrido (170 016 filas, 7,6 MB) no se versiona; lo regenera
`scripts/sweep.py` por semilla global. Las tablas de las que salen todas las
cifras del texto sí están, en `artifacts/tables/`.

## Reproducir

Resultados obtenidos con Python 3.11.14, PyTorch 2.11.0+cu130, CUDA 13.0,
sobre una NVIDIA GeForce RTX 5090. El barrido es apto para CPU; solo los
cuatro forwards de la validación end-to-end piden GPU.

Instalación **aditiva**, que no barre el entorno:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
```

Los pesos de ViT-B y el conjunto de validación de ImageNet-100 no son
redistribuibles: se descargan aparte y sus rutas se declaran en
`configs/checkpoints.local.yaml`, que no se versiona, nunca en el código.
Pythia-410M se toma de sus pesos públicos.

```yaml
# configs/checkpoints.local.yaml
vitb_ckpt: /ruta/al/checkpoint_seed42.pt
imagenet100_root: /ruta/al/val/de/imagenet-100
```

Los valores admiten variables de entorno (`$VAR`). Sin ese fichero, los
pasos que no necesitan esas rutas siguen corriendo; los que sí, avisan de
cuál falta.

```bash
python scripts/sanity.py          # las tres compuertas previas
python scripts/sweep.py           # el barrido completo
python scripts/lectura_fria.py    # Q1, Q2, Q3 y la cola GL
python scripts/validacion_e2e.py  # los cuatro forwards
python scripts/figuras.py         # las seis figuras
```

## Afirmación → script → dato

| Afirmación, tabla o figura | Script | Dato |
|---|---|---|
| Las tres compuertas previas (suelo fp64, int8 ≪ int4, idempotencia) | `scripts/sanity.py` | salida en consola |
| El barrido completo (170 016 medidas) | `scripts/sweep.py` | `artifacts/logs/quant_orbita/quant_orbita.csv` |
| Q1, anchura de la órbita ortogonal | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q1.csv` |
| Q2, distancia del gauge entrenado al suelo | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q2.csv` |
| Q3, por-cabeza frente a compartida | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q3.csv` |
| La cola GL por percentiles | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_gl.csv` |
| Tabla 1, validación end-to-end | `scripts/validacion_e2e.py` | `artifacts/tables/validacion_e2e.csv` |
| Figuras 1 a 3, por modelo | `scripts/figuras.py` | `artifacts/figs/*.png` |

Cualquier gauge muestreado se reconstruye exacto reproduciendo la secuencia
del generador que el barrido consumió, sin persistir las matrices
(`src/reconstruccion.py`).

## Cómo citar

```bibtex
@unpublished{munozpla2026plana,
  author = {Muñoz Plá, Manuel},
  title  = {Flat inside, toxic outside: the value-output gauge orbit
            under quantization},
  year   = {2026},
  note   = {Manuscrito},
  url    = {https://github.com/mmunozpl/gauge-orbit-quantization}
}
```

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
