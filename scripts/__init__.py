import numpy as np
from yaml import add_representer
from itertools import product
from dagflow.bundles.load_hist import load_hist
from dagflow.parameters import Parameter
from dagflow.core import NodeStorage


add_representer(
    np.ndarray,
    lambda representer, obj: representer.represent_str(np.array_repr(obj)),
)


def update_dict_parameters(
    dict_parameters: dict[str, Parameter], groups: list[str], model_parameters: NodeStorage,
) -> None:
    """Update dictionary of minimization parameters

    Parameters
    ----------
        dict_parameters : dict[str, Parameter])
            dictionary of parameters
        groups : list[str]
            list of groups of parameters to be added to dict_parameters
        model_parameters : NodeStorage
            storage of model parameters

    Returns
    -------
    None
    """
    for group in groups:
        dict_parameters.update(dict(model_parameters(group).walkjoineditems()))


def load_model_from_file(filename: str, node_name: str, name_pattern: str, groups: list[str]):
    """Update dictionary of minimization parameters

    Parameters
    ----------
        filename : str
            path to file that contains model observations
        node_name : str
            name of node where outputs model observations will be stored
        name_pattern : str
            pattern that determines model observations in file,
            pattern uses two placeholders: for detector and for item from `groups`
        groups : list[str]
            list of groups to be added to NodeStorage

    Returns
    -------
    NodeStorage
        Storage that contains model observations
    """
    comparison_storage = load_hist(
        name=node_name,
        x="erec",
        y="fine",
        merge_x=True,
        filenames=filename,
        replicate_outputs=list(product(["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"], groups)),
        skip=({"AD22", "6AD"}, {"AD34", "6AD"}, {"AD11", "7AD"}),
        name_function=lambda _, idx: name_pattern.format(*idx)
    )
    return comparison_storage["outputs"]


def filter_fit(src: dict, keys_to_fiter: list[str]) -> None:
    """Remove keys from fit dictionary

    Parameters
    ----------
        src : dict
            dictionary of fit
        keys_to_filter : list[str]
            list of keys to be deleted from fit dictionary

    Returns
    -------
    None
    """
    keys = list(src.keys())
    for key in keys:
        if key in keys_to_fiter:
            del src[key]
            continue
        if isinstance(src[key], dict):
            filter_fit(src[key], keys_to_fiter)


def convert_numpy_to_lists(src: dict) -> None:
    """Convert recursively numpy array in dictionary

    Parameters
    ----------
        src : dict
            dictionary that may contains numpy arrays as value

    Returns
    -------
    None
    """
    for key, value in src.items():
        if isinstance(value, np.ndarray):
            src[key] = value.tolist()
        elif isinstance(value, dict):
            convert_numpy_to_lists(value)

