# [Preprint] Gauge equivalence does not survive quantisation: value-output orbit, norm-product bound, and consequences for rotational methods

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá** · [ORCID 0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22904207-009e73)](https://doi.org/10.5281/zenodo.22904207)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Dataset-ffd21e)](https://huggingface.co/datasets/ManPla/gauge-orbit-quantization-results)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-ffd21e)](https://huggingface.co/spaces/ManPla/gauge-orbit-quantization-demo)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5714--912X-a6ce39)](https://orcid.org/0009-0000-5714-912X)
[![Web](https://img.shields.io/badge/Web-manpla.net-009e73)](https://manpla.net)
[![License](https://img.shields.io/badge/License-Apache--2.0-009e73)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#how-to-cite)

**Abstract:** Rotational quantisation methods exploit a computational
invariance of attention's value-output sector: rotating the pair
(*W_v*, *W_O*) without altering the function. That invariance is a subgroup
of a larger orbit —*GL(d_h)* per head, certified in the previous work— and
practice restricts it twice: to orthogonal rotations, and shared across the
heads of a layer. Here we measure what both restrictions buy in terms of
the OV circuit's quantisation error, on two architectures (Pythia-410M and a
fine-tuned ViT-B/16), with a deliberately simple quantiser that isolates the
variable of interest. Three results. Within the orthogonal subgroup the orbit
is flat: the error range across rotations stays at ×1.005–1.02 in median, and
the trained gauge is already within ×1.02–1.04 of the best sampled point.
Per-head freedom improves on the shared rotation by 1.3 per cent, below the
threshold the design fixed as material. And in the sampled GL family, increasing
anisotropy raises the error monotonically, up to ×15 in median and ×10⁵ at
the extreme. The mechanism admits a closed form: the
error is bounded by the product of the transformed factors' norms, and the
approximation that follows from the bound predicts exponent one, with 0.998
measured across two gauge families, so that the catastrophic tail turns out
to be the dispersion of conditioning transmitted by that relation. The bound also names the design space's border,
which does not separate orthogonal from non-orthogonal: the conformal
orthogonal class leaves the bound's factor p invariant by identity, the
balanced point —non-orthogonal—
minimises the bound over the entire orbit, and the trained gauge sits within
1.1–1.5 per cent of that optimum. An end-to-end validation confirms the
flatness in real accuracy: identity, best and worst sampled rotation and
balanced point fall within 0.10 percentage points, indistinguishable by
paired McNemar over the same 5,000 images. The decisive contrast is a
different one. A general gauge at scale 2, unquantised, predicts
the same class as the identity on all 5,000 images, without a single
discrepancy. Quantised to int4 it falls to 0.0082, below nominal chance
over 100 classes, and then agrees with the identity on 44 images. The control isolates the cause: gauge equivalence is exact
in real arithmetic and does not survive the grid. The result further delimits
where the gain of rotational methods does not lie —not in the orbit's weight
landscape—, consistent with their usual attribution to activation outliers.

**Keywords:** quantisation, computational invariance, gauge orbit, OV
circuit, multi-head attention, RTN.

## Contents

```
.
├── src/          # fp64 gauges, RTN quantiser, circuit error, per-head
│                 # weights and reconstruction by generator replay
├── scripts/      # sanity gates, sweep, cold reading, end-to-end
│                 # validation and figures
├── configs/      # paths to models and data, never in the code
├── artifacts/    # reading tables and figures
└── paper/        # both PDFs; the .tex source is not versioned
```

The raw csvs are not versioned, for their size —170,016 rows for the regime
sweep, 642,048 for the controlled-conditioning one and 102,432 for the
re-sweep of the original family with norms—; their scripts regenerate them
from the global seed. The tables every figure in the text comes from are
versioned, under `artifacts/tables/`.

## Reproducing

Results obtained with Python 3.11.14, PyTorch 2.11.0+cu130, CUDA 13.0, on an
NVIDIA GeForce RTX 5090. The sweep is CPU-friendly; only the four forwards of
the end-to-end validation need a GPU.

**Additive** installation, which does not wipe the environment:

```bash
uv pip install -r <(uv export --no-hashes --no-dev)
```

The ViT-B weights and the ImageNet-100 validation set are not
redistributable: they are downloaded separately and their paths declared in
`configs/checkpoints.local.yaml`, which is not versioned, never in the code.
Pythia-410M is taken from its public weights.

```yaml
# configs/checkpoints.local.yaml
vitb_ckpt: /path/to/checkpoint_seed42.pt
imagenet100_root: /path/to/imagenet-100/val
```

Values accept environment variables (`$VAR`). Without that file, the steps
that do not need those paths still run; the ones that do report which is
missing.

```bash
pytest -v tests/                    # the four gates
python scripts/sweep.py             # regime sweep
python scripts/lectura_fria.py      # Q1, Q2, Q3 and the GL tail
python scripts/validacion_e2e.py    # the four forwards
python scripts/figuras.py           # the first six figures
python scripts/sweep_kappa.py       # controlled conditioning
python scripts/sweep_familia.py     # original family, with norms
python scripts/lectura_kappa.py     # the preregistered rule on κ
python scripts/colapso_producto.py  # the product law
python scripts/figuras_kappa.py     # the six new figures
python scripts/punto_balanceado.py --modelo pythia   # the bound, tested
python scripts/punto_balanceado.py --modelo vitb     # from the other side
python scripts/contraste_e2e.py     # paired McNemar and bootstrap
```

The interactive demo runs locally and does not need the original models.
`scripts/extraer_sector_vo.py` builds a light carrier of the value-output
sector, `demo/app.py` serves it with Gradio, and `scripts/paridad_demo.py`
certifies cell by cell that the demo returns the same numbers as the
published sweeps. The app runs `src/` unchanged and does not reimplement
the arithmetic.

```bash
python scripts/extraer_sector_vo.py --modelo vitb
python scripts/extraer_sector_vo.py --modelo pythia
python scripts/paridad_demo.py      # the demo's three gates
python demo/app.py                  # needs gradio, in its own venv
```

The tests that need the ViT-B weights are skipped without them, with the
reason stated, rather than failing. The κ, conformal-class and balanced-point
gates need no data and always run.

## Claim → script → data

| Claim, table or figure | Script | Data |
|---|---|---|
| The three prior gates (fp64 floor, int8 ≪ int4, idempotence) | `scripts/sanity.py` | console output |
| The full sweep (170,016 measurements) | `scripts/sweep.py` | `artifacts/logs/quant_orbita/quant_orbita.csv` |
| Q1, width of the orthogonal orbit | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q1.csv` |
| Q2, distance from the trained gauge to the floor | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q2.csv` |
| Q3, per-head against shared | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_q3.csv` |
| The GL tail by percentiles | `scripts/lectura_fria.py` | `artifacts/tables/lectura_fria_{pythia,vitb}_gl.csv` |
| Table 1, end-to-end validation | `scripts/validacion_e2e.py` | `artifacts/tables/validacion_e2e.csv` |
| Figures 1 to 3, per model | `scripts/figuras.py` | `artifacts/figs/{pythia,vitb}_{regimenes,cola_gl,por_cabeza_vs_compartida}_bits4.png` |
| Controlled-conditioning sweep (642,048 measurements) | `scripts/sweep_kappa.py` | `artifacts/logs/quant_kappa/quant_kappa.csv` |
| Re-sweep of the original family, with norms | `scripts/sweep_familia.py` | `artifacts/logs/quant_kappa/quant_familia.csv` |
| Verdict on κ (trend, not law) | `scripts/lectura_kappa.py` | `artifacts/tables/lectura_kappa.csv` |
| The product law and the between-family collapse | `scripts/colapso_producto.py` | `artifacts/tables/colapso_producto.csv` |
| Between-family consistency at matched κ | `scripts/colapso_producto.py` | `artifacts/tables/consistencia_kappa.csv` |
| κ and collapse figures | `scripts/figuras_kappa.py` | `artifacts/figs/{pythia,vitb}_kappa_bits{4,8}.png`, `colapso_producto_bits{4,8}.png` |
| The balanced point, the bound tested out of sample (1,056 measurements) | `scripts/punto_balanceado.py` | `artifacts/tables/punto_balanceado.csv` |
| Paired contrast of the end-to-end validation | `scripts/contraste_e2e.py` | `artifacts/tables/contraste_e2e.csv` |

Any sampled gauge is reconstructed exactly by replaying the generator
sequence the sweep consumed, with no need to persist the matrices
(`src/reconstruccion.py`).

## How to cite

```bibtex
@software{munozpla2026gaugeequivalence,
  author  = {Muñoz Plá, Manuel},
  title   = {Gauge equivalence does not survive quantisation:
             value-output orbit, norm-product bound, and consequences
             for rotational methods},
  year    = {2026},
  version = {v1.0.1},
  doi     = {10.5281/zenodo.22904207},
  url     = {https://github.com/mmunozpl/gauge-orbit-quantization}
}
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
