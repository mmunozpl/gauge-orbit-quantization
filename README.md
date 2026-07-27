# The gauge orbit as a design space for quantization

Short note (in progress). Companion piece to *Same function, different
pruning: the dominant direction of $W_O$ under free, soft, and hard
intervention* (paper 1, <https://github.com/mmunozpl/angular-separation-vit>),
which certifies the value-output gauge orbit ($GL(d_h)$ per head) as
the space where $v_1(W_O)$ moves without touching the function.
QuaRot/SpinQuant already exploit a subgroup of that orbit (orthogonal,
per-layer-shared rotations) to kill outliers before quantizing. This
note measures what that practice assumes:

- **Q1** — how wide is the orbit's quantization-error range for the OV
  circuit?
- **Q2** — how far is the trained (identity) gauge from the orbit's
  optimum?
- **Q3** — what does per-head rotation freedom buy over the
  layer-shared rotation SpinQuant uses?

Status: design and pre-registered readings fixed; sweep not yet run.
See the project's own planning notes (not versioned here) for the
full specification.

## Requirements

Same environment as paper 1: `conda activate pytorch28`, Python
≥ 3.11, PyTorch, `timm`, `transformers`. No new dependencies — the
RTN quantizer is self-contained project code.

## Layout

```
src/       # RTN quantizer, gauge construction, OV-circuit error
scripts/   # sweep entry points, per-model
configs/   # sweep configs
tests/     # pytest suite
```

## License

Apache-2.0. See [LICENSE](LICENSE).
