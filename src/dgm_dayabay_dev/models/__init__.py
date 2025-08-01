from typing import Mapping

from dag_modelling.tools.logger import logger

from .dayabay_v0 import model_dayabay_v0
from .dayabay_v0b import model_dayabay_v0b
from .dayabay_v0c import model_dayabay_v0c
from .dayabay_v0d import model_dayabay_v0d
from .dayabay_v0e import model_dayabay_v0e
from .dayabay_v0f import model_dayabay_v0f

from .dayabay_labels import LATEX_SYMBOLS


AD_TO_EH = {
    "AD11": "EH1",
    "AD12": "EH1",
    "AD21": "EH2",
    "AD22": "EH2",
    "AD31": "EH3",
    "AD32": "EH3",
    "AD33": "EH3",
    "AD34": "EH3",
}

_dayabay_models = {
    "v0": model_dayabay_v0,
    "v0b": model_dayabay_v0b,
    "v0c": model_dayabay_v0c,
    "v0d": model_dayabay_v0d,
    "v0e": model_dayabay_v0e,
    "v0f": model_dayabay_v0f,
}
_dayabay_models["latest"] = _dayabay_models["v0e"]

_available_sources = ("tsv", "hdf5", "root", "npz")


def available_models() -> tuple[str, ...]:
    return tuple(_dayabay_models.keys())

def available_sources() -> tuple[str, ...]:
    return _available_sources

def load_model(version, model_options: Mapping | str = {}, **kwargs):
    if isinstance(model_options, str):
        from yaml import Loader, load

        model_options = load(model_options, Loader)

    if not isinstance(model_options, dict):
        raise RuntimeError(
            "model_options expects a python dictionary or yaml dictionary"
        )

    model_options = dict(model_options, **kwargs)

    logger.info(f"Execute Daya Bay model {version}")
    try:
        cls = _dayabay_models[version]
    except KeyError:
        raise RuntimeError(
            f"Invalid model version {version}. Available models: {', '.join(sorted(_dayabay_models.keys()))}"
        )

    return cls(**model_options)
