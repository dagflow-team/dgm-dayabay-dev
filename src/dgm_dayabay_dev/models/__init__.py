from typing import Mapping

from dag_modelling.tools.logger import logger

from .dayabay_labels import LATEX_SYMBOLS
from .dayabay_v0 import model_dayabay_v0
from .dayabay_v0b import model_dayabay_v0b
from .dayabay_v0c import model_dayabay_v0c
from .dayabay_v0d import model_dayabay_v0d
from .dayabay_v0e import model_dayabay_v0e
from .dayabay_v0f import model_dayabay_v0f
from .dayabay_v1 import model_dayabay_v1
from .dayabay_v1a import model_dayabay_v1a
from .dayabay_v1a_distorted import model_dayabay_v1a_distorted
from .dayabay_v1a_neutrino_rate.model_dayabay import model_dayabay as model_dayabay_v1a_neutrino_rate

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
    "v1": model_dayabay_v1,
    "v1a": model_dayabay_v1a,
    "v1a_distorted": model_dayabay_v1a_distorted,
    "v1a_neutrino_rate": model_dayabay_v1a_neutrino_rate,
}
_dayabay_models["latest"] = _dayabay_models["v1a"]

_available_sources = ("tsv", "hdf5", "root", "npz")


def available_models() -> tuple[str, ...]:
    return tuple(_dayabay_models.keys())


def available_models_limited(*, first: str, last: str | None = None) -> tuple[str, ...]:
    names = tuple(_dayabay_models.keys())
    assert names
    i = 0
    for i, name in enumerate(names):
        if name == first:
            break

    if last is None:
        return names[i:]

    ret = []
    for i in range(i, len(names)):
        ret.append(name := names[i])
        if name == last:
            break

    return tuple(ret)


def available_sources() -> tuple[str, ...]:
    return _available_sources


def load_model(version, model_options: Mapping | str = {}, **kwargs):
    if isinstance(model_options, str):
        from yaml import Loader, load

        model_options = load(model_options, Loader)

    if not isinstance(model_options, dict):
        raise RuntimeError("model_options expects a python dictionary or yaml dictionary")

    model_options = dict(model_options, **kwargs)

    logger.info(f"Execute Daya Bay model {version}")
    try:
        cls = _dayabay_models[version]
    except KeyError:
        raise RuntimeError(
            f"Invalid model version {version}. Available models: {', '.join(sorted(_dayabay_models.keys()))}"
        )

    return cls(**model_options)
