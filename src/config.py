"""cargador único de configuraciones YAML."""

import os
from pathlib import Path

import yaml


class _Config(dict):
    """configuración que exige valor a las rutas sin declarar."""

    def __getitem__(self, clave: str) -> object:
        """devuelve el valor de una clave, o explica que falta.

        args:
            clave: nombre de la entrada de configuración.

        returns:
            el valor asociado a la clave.

        raises:
            ValueError: si la clave existe pero está sin declarar.
        """
        valor = super().__getitem__(clave)
        if valor in (None, ""):
            raise ValueError(
                f"«{clave}» está sin declarar: es una ruta del árbol "
                f"propio y no se versiona. Se declara en el fichero "
                f"«.local.yaml» que acompaña a la configuración, o por "
                f"variable de entorno. Ver LEEME.md."
            )
        return valor


def _lee(ruta: Path) -> dict:
    """lee un yaml y devuelve su contenido, vacío si no trae nada.

    args:
        ruta: ruta al fichero .yaml.

    returns:
        diccionario con el contenido del yaml.
    """
    with open(ruta, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _expande(valor: str | None) -> str | None:
    """expande variables de entorno en un valor de texto.

    args:
        valor: entrada de la configuración.

    returns:
        el valor con las variables de entorno ya sustituidas.
    """
    return os.path.expandvars(valor) if isinstance(valor, str) else valor


def load_config(path: str) -> dict:
    """lee un yaml y le superpone su variante local, si existe.

    las rutas que apuntan al árbol propio no se versionan: las aporta
    el fichero «<nombre>.local.yaml» que acompaña al de la
    configuración, con prioridad sobre él.

    args:
        path: ruta al fichero .yaml.

    returns:
        configuración ya superpuesta y con las variables de entorno
        expandidas.
    """
    base = Path(path)
    cfg = _lee(base)
    local = base.with_suffix(".local.yaml")
    if local.exists():
        cfg.update(_lee(local))
    return _Config({k: _expande(v) for k, v in cfg.items()})
