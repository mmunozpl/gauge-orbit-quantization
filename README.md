# [Preprint] Flat inside, toxic outside: the value-output gauge orbit under quantization

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá** · [ORCID 0009-0000-5714-912X](https://orcid.org/0009-0000-5714-912X)

[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--5714--912X-a6ce39)](https://orcid.org/0009-0000-5714-912X)
[![License](https://img.shields.io/badge/License-Apache--2.0-009e73)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#how-to-cite)

**Abstract:** Rotational quantization methods exploit a computational
invariance of attention's value-output sector: rotating the pair
(*W_v*, *W_O*) without altering the function. That invariance is a subgroup
of a larger orbit —*GL(d_h)* per head, certified in the previous work— and
practice restricts it twice: to orthogonal rotations, and shared across the
heads of a layer. This note measures what both restrictions buy in terms of
the OV circuit's quantization error, on two architectures (Pythia-410M and a
fine-tuned ViT-B/16), with a deliberately simple quantizer that isolates the
variable of interest. Three results. Within the orthogonal subgroup the orbit
is flat: the error range across rotations stays at ×1.01–1.02 in median, and
the trained gauge is already within ×1.02–1.04 of the best sampled point.
Per-head freedom does not improve on the shared rotation (median improvement
≈ 0 per cent). And outside the subgroup, the general orbit degrades the error
monotonically with gauge strength, up to ×15 in median and ×10⁵ at the
extreme. The conclusion is asymmetric: practice's orthogonal restriction is
necessary, not a convenience; the point chosen within it is nearly
indifferent for the weights, at these scales. An end-to-end validation
confirms the orthogonal flatness in real accuracy —the best and worst sampled
rotation give an indistinguishable top-1 from each other, with a range of
0.08 percentage points—, though it also reveals, as an unpreregistered
observation, that the identity retains a small and consistent advantage over
any orthogonal gauge (≈ 2.8 points), a signal that the circuit error does not
tell the whole story. More striking still: a gauge outside the subgroup, with
the exact circuit mathematically intact in fp64, collapses accuracy to chance
level (1.00 per cent over 100 classes) —quantization turns a choice of
coordinates into a random classifier—. The result further bounds where the
gain of rotational methods does not lie —not in the orbit's weight
landscape—, consistent with their usual attribution to activation outliers.

**Keywords:** quantization, computational invariance, gauge orbit, OV
circuit, multi-head attention, RTN.

## Contents

```
.
├── src/          # fp64 gauges, RTN quantizer, circuit error, per-head
│                 # weights and reconstruction by generator replay
├── scripts/      # sanity gates, sweep, cold reading, end-to-end
│                 # validation and figures
├── configs/      # paths to models and data, never in the code
└── artifacts/    # reading tables and figures
```

The sweep's raw csv (170,016 rows, 7.6 MB) is not versioned; `scripts/sweep.py`
regenerates it from the global seed. The tables every figure in the text comes
from are versioned, under `artifacts/tables/`.

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
python scripts/sanity.py          # the three prior gates
python scripts/sweep.py           # the full sweep
python scripts/lectura_fria.py    # Q1, Q2, Q3 and the GL tail
python scripts/validacion_e2e.py  # the four forwards
python scripts/figuras.py         # the six figures
```

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
| Figures 1 to 3, per model | `scripts/figuras.py` | `artifacts/figs/*.png` |

Any sampled gauge is reconstructed exactly by replaying the generator
sequence the sweep consumed, with no need to persist the matrices
(`src/reconstruccion.py`).

## How to cite

```bibtex
@unpublished{munozpla2026plana,
  author = {Muñoz Plá, Manuel},
  title  = {Flat inside, toxic outside: the value-output gauge orbit
            under quantization},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/mmunozpl/gauge-orbit-quantization}
}
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
