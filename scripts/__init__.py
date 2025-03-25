from dagflow.parameters import GaussianParameter
from dagflow.core import NodeStorage


def update_dict_parameters(
    dict_parameters: dict[str, GaussianParameter], groups: list[str], model_parameters: NodeStorage,
) -> None:
    """Update dictionary of minimization parameters

    Parameters
    ---------
        dict_parameters : dict[str, Parameter])
            dictionary of parameters
        groups : list[str]
            list of groups of parameters to be added to dict_parameters
        model_parameters : NodeStorage
            storage of model parameters
    """
    for group in groups:
        dict_parameters.update({".".join([group, *tuple_path]): parameter for tuple_path, parameter in model_parameters(group).walkitems()})

