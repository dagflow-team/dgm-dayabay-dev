from dagflow.parameters import Parameter
from dagflow.core import NodeStorage


def update_dict_parameters(
    dict_parameters: dict[str, Parameter], groups: list[str], model_parameters: NodeStorage,
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
        dict_parameters.update(dict(model_parameters(group).walkjoineditems()))

