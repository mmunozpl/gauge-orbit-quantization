"""carga de modelos y extracción por-cabeza de (w_v, w_o).

reimplementación autocontenida, para este repo, de la extracción ya
verificada en el paper 1 (angular-separation-vit): gpt-neox fusiona
q,k,v por cabeza intercalado (``query_key_value``); timm concatena
[q,k,v] por bloque y v ocupa el último tercio (``qkv``). convención
de forma en todo el repo: w_v^(h) es [d_h, d], w_o^(h) es [d, d_h].
"""

import torch


def cargar_pythia(
    model_id: str = "EleutherAI/pythia-410m",
    dtype: torch.dtype = torch.float32,
):
    """carga pythia-410m y verifica su anatomía contra el config real.

    args:
        model_id: identificador del modelo en hugging face hub.
        dtype: precisión de carga.

    returns:
        el modelo en modo evaluación.
    """
    from transformers import AutoModelForCausalLM

    modelo = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager")
    modelo.eval()
    cfg = modelo.config
    dim_cabeza = cfg.hidden_size // cfg.num_attention_heads
    assert dim_cabeza == 64, f"dim_cabeza inesperada: {dim_cabeza}"
    assert cfg.num_attention_heads == 16, "num_attention_heads != 16"
    assert cfg.hidden_size == 1024, "hidden_size != 1024"
    assert len(modelo.gpt_neox.layers) == 24, "num capas != 24"
    return modelo


@torch.no_grad()
def w_v_w_o_pythia(
    modelo,
    capa: int,
    n_cabezas: int = 16,
    dim_cabeza: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(w_v, w_o) por cabeza de una capa de pythia, en float64.

    query_key_value fusiona q, k y v por cabeza: cada cabeza ocupa
    ``3*dim_cabeza`` filas contiguas ``[q_h|k_h|v_h]``; v_h es el
    último tercio. dense es la proyección de salida (columnas por
    cabeza).

    args:
        modelo: modelo gpt-neox (pythia) cargado.
        capa: índice de capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        (w_v [h, d_h, d], w_o [h, d, d_h]), float64.
    """
    attn = modelo.gpt_neox.layers[capa].attention
    w_qkv = attn.query_key_value.weight  # [3d, d]
    w_dense = attn.dense.weight          # [d, d]
    w_v, w_o = [], []
    for h in range(n_cabezas):
        base = h * 3 * dim_cabeza + 2 * dim_cabeza
        w_v.append(w_qkv[base:base + dim_cabeza, :].double())
        w_o.append(w_dense[:, h * dim_cabeza:(h + 1) * dim_cabeza]
                   .double())
    return torch.stack(w_v), torch.stack(w_o)


def cargar_vitb(
    ckpt_path: str,
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 100,
    img_size: int = 224,
):
    """carga la columna vit-b con los pesos de un checkpoint de a.

    args:
        ckpt_path: ruta al checkpoint (.pt con clave 'model',
            prefijo 'backbone.' en las claves).
        model_name: identificador timm de la columna.
        num_classes: clases de la cabeza.
        img_size: resolución de entrada.

    returns:
        el vit interno de timm (el que expone .blocks), en eval.
    """
    import timm

    modelo = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes,
        img_size=img_size)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # el checkpoint viene de un wrapper HeadProjections del paper 1:
    # las claves llevan 'backbone.model.' antes de la state_dict plana
    # de timm ('backbone.' del checkpoint de entrenamiento, 'model.'
    # del atributo submódulo del wrapper).
    sd = {k.replace("backbone.model.", "", 1): v
          for k, v in blob["model"].items()}
    modelo.load_state_dict(sd)
    return modelo.eval()


@torch.no_grad()
def w_v_w_o_vitb(
    modelo,
    capa: int,
    n_cabezas: int = 12,
    dim_cabeza: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(w_v, w_o) por cabeza de una capa de vit-b, en float64.

    en timm el qkv concatena [q, k, v] por bloque; v ocupa el último
    tercio y se reordena por cabeza. proj es la proyección de salida.

    args:
        modelo: el vit interno de timm.
        capa: índice de capa.
        n_cabezas: cabezas h.
        dim_cabeza: d_h.

    returns:
        (w_v [h, d_h, d], w_o [h, d, d_h]), float64.
    """
    attn = modelo.blocks[capa].attn
    w_qkv = attn.qkv.weight    # [3d, d]
    w_proj = attn.proj.weight  # [d, d]
    base_v = 2 * w_qkv.shape[1]
    w_v, w_o = [], []
    for h in range(n_cabezas):
        fil = slice(base_v + h * dim_cabeza, base_v + (h + 1) * dim_cabeza)
        col = slice(h * dim_cabeza, (h + 1) * dim_cabeza)
        w_v.append(w_qkv[fil, :].double())
        w_o.append(w_proj[:, col].double())
    return torch.stack(w_v), torch.stack(w_o)


@torch.no_grad()
def escribe_v_o_vitb(
    modelo,
    capa: int,
    h: int,
    w_v_h: torch.Tensor,
    w_o_h: torch.Tensor,
    dim_cabeza: int = 64,
) -> None:
    """escribe (w_v, w_o) de una cabeza de vuelta en el modelo, in situ.

    inversa de `w_v_w_o_vitb` para una sola cabeza: usada por la
    validación end-to-end para materializar un modelo con los pesos
    ya gaugeados-y-cuantizados antes del forward real.

    args:
        modelo: el vit interno de timm, modificado in situ.
        capa: índice de capa.
        h: índice de cabeza.
        w_v_h: tensor [d_h, d] (cualquier dtype, se castea al del
            modelo).
        w_o_h: tensor [d, d_h].
        dim_cabeza: d_h.
    """
    attn = modelo.blocks[capa].attn
    w_qkv = attn.qkv.weight
    w_proj = attn.proj.weight
    base_v = 2 * w_qkv.shape[1]
    fil = slice(base_v + h * dim_cabeza, base_v + (h + 1) * dim_cabeza)
    col = slice(h * dim_cabeza, (h + 1) * dim_cabeza)
    w_qkv[fil, :] = w_v_h.to(w_qkv.dtype)
    w_proj[:, col] = w_o_h.to(w_proj.dtype)
