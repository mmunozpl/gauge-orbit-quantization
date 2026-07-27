"""cargador único de configuraciones YAML."""

import yaml


def load_config(path: str) -> dict:
    """lee un archivo yaml y devuelve su contenido.

    args:
        path: ruta al fichero .yaml.

    returns:
        diccionario con el contenido del yaml.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
