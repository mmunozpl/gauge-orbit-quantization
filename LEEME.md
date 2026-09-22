# [Preprint] La equivalencia de gauge no sobrevive a la cuantización: órbita valor-salida, cota del producto de normas y consecuencias para los métodos rotacionales

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
×1,005–1,02 en mediana, y el gauge entrenado ya está a ×1,02–1,04 del mejor
punto muestreado. La libertad por-cabeza mejora en un 1,3 % a la rotación
compartida, por debajo del listón que el diseño fijó como material. Y en la familia
GL muestreada, aumentar la anisotropía eleva el error de forma monótona,
hasta ×15 en mediana y ×10⁵ en el extremo. El mecanismo
admite forma cerrada: el error queda acotado por el producto de las normas de
los factores transformados, y la aproximación que se sigue de la cota predice
exponente uno, con 0,998 medido sobre dos familias de gauge, de modo que la
cola catastrófica resulta ser la dispersión del condicionamiento transmitida
por esa relación. La cota nombra
además la frontera del espacio de diseño, que no separa lo ortogonal de lo no
ortogonal: la clase conforme ortogonal deja invariante por identidad el
factor p de la cota, el punto
balanceado —no ortogonal— minimiza la cota sobre la órbita entera, y el gauge
entrenado queda a un 1,1–1,5 % de ese óptimo. Una validación end-to-end
confirma la llanura en exactitud real: identidad, mejor y peor rotación
muestreada y punto balanceado caen en 0,10 puntos porcentuales,
indistinguibles por McNemar pareado sobre las mismas 5000 imágenes. El
contraste que carga el peso es otro. Un gauge general de escala 2, sin
cuantizar, predice la misma clase que la identidad en las 5000 imágenes,
sin una sola discrepancia. Cuantizado a int4 cae a 0,0082, por debajo del
azar nominal sobre 100 clases, y coincide entonces con la identidad en 44
imágenes. El control aísla la causa: la
equivalencia de gauge es exacta en aritmética real y no sobrevive a la
rejilla. El resultado acota además dónde no está la ganancia de los métodos
rotacionales —no en el paisaje de pesos de la órbita—, coherente con su
atribución habitual a los outliers de activación.

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
├── artifacts/    # tablas de lectura y figuras
└── paper/        # los dos PDF; el fuente .tex no se versiona
```

Los csv crudos no se versionan por peso —170 016 filas del barrido de
regímenes, 642 048 del de condicionamiento controlado y 102 432 del re-barrido
de la familia original con normas—; los regeneran sus scripts por semilla
global. Las tablas de las que salen todas las cifras del texto sí están, en
`artifacts/tables/`.

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
pytest -v tests/                    # las cuatro compuertas
python scripts/sweep.py             # barrido de regímenes
python scripts/lectura_fria.py      # Q1, Q2, Q3 y la cola GL
python scripts/validacion_e2e.py    # los cuatro forwards
python scripts/figuras.py           # las seis primeras figuras
python scripts/sweep_kappa.py       # condicionamiento controlado
python scripts/sweep_familia.py     # familia original, con normas
python scripts/lectura_kappa.py     # la regla preregistrada sobre κ
python scripts/colapso_producto.py  # la ley del producto
python scripts/figuras_kappa.py     # las seis figuras nuevas
python scripts/punto_balanceado.py --modelo pythia   # la cota, probada
python scripts/punto_balanceado.py --modelo vitb     # por el otro lado
python scripts/contraste_e2e.py     # McNemar y bootstrap pareados
```

La demo interactiva corre en local y no necesita los modelos originales.
`scripts/extraer_sector_vo.py` construye un portador ligero del sector
valor-salida, `demo/app.py` lo sirve con Gradio, y `scripts/paridad_demo.py`
certifica celda a celda que la demo devuelve los mismos números que los
barridos publicados. La aplicación ejecuta `src/` sin cambios y no
reimplementa la aritmética.

```bash
python scripts/extraer_sector_vo.py --modelo vitb
python scripts/extraer_sector_vo.py --modelo pythia
python scripts/paridad_demo.py      # las tres compuertas de la demo
python demo/app.py                  # necesita gradio, en su entorno
```

Los tests que necesitan los pesos del ViT-B se saltan sin ellos, con el motivo
escrito, en vez de fallar. Las compuertas de κ, de la clase conforme y del
punto balanceado no necesitan datos y corren siempre.

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
| Figuras 1 a 3, por modelo | `scripts/figuras.py` | `artifacts/figs/{pythia,vitb}_{regimenes,cola_gl,por_cabeza_vs_compartida}_bits4.png` |
| Barrido de condicionamiento controlado (642 048 medidas) | `scripts/sweep_kappa.py` | `artifacts/logs/quant_kappa/quant_kappa.csv` |
| Re-barrido de la familia original, con normas | `scripts/sweep_familia.py` | `artifacts/logs/quant_kappa/quant_familia.csv` |
| Veredicto sobre κ (tendencia, no ley) | `scripts/lectura_kappa.py` | `artifacts/tables/lectura_kappa.csv` |
| La ley del producto y el colapso entre familias | `scripts/colapso_producto.py` | `artifacts/tables/colapso_producto.csv` |
| Consistencia entre familias a κ igualado | `scripts/colapso_producto.py` | `artifacts/tables/consistencia_kappa.csv` |
| Figuras de κ y del colapso | `scripts/figuras_kappa.py` | `artifacts/figs/{pythia,vitb}_kappa_bits{4,8}.png`, `colapso_producto_bits{4,8}.png` |
| El punto balanceado, la cota probada fuera de muestra (1056 medidas) | `scripts/punto_balanceado.py` | `artifacts/tables/punto_balanceado.csv` |
| Contraste pareado de la validación end-to-end | `scripts/contraste_e2e.py` | `artifacts/tables/contraste_e2e.csv` |

Cualquier gauge muestreado se reconstruye exacto reproduciendo la secuencia
del generador que el barrido consumió, sin persistir las matrices
(`src/reconstruccion.py`).

## Cómo citar

```bibtex
@unpublished{munozpla2026gaugeequivalence,
  author = {Muñoz Plá, Manuel},
  title  = {Gauge equivalence does not survive quantisation:
            value-output orbit, norm-product bound, and consequences
            for rotational methods},
  year   = {2026},
  note   = {Manuscrito},
  url    = {https://github.com/mmunozpl/gauge-orbit-quantization}
}
```

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
