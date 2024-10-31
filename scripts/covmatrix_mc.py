#!/usr/bin/env python
import numpy as np
from argparse import Namespace
from matplotlib import pyplot as plt

from dagflow.tools.logger import DEBUG as INFO4
from dagflow.tools.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

from dagflow.core import NodeStorage
from dagflow.core.output import Output
from dagflow.parameters import Parameter
from multikeydict.nestedmkdict import walkvalues
from multikeydict.typing import properkey
from numpy.typing import NDArray


set_level(INFO1)


def variate_parameters(parameters: list[Parameter], generator: np.random.Generator) -> None:
    """Randomize value of parameters via normal unit distribution N(0, 1)

        Parameters
        ----------
        parameters: list[Parameter]
            list of normalized parameters
        generator: np.random.Generator
            numpy generator of pseudo-random numbers

        Returns
        -------
        None
    """
    for parameter in parameters:
        parameter.value = generator.normal(0, 1)


def create_list_of_variation_parameters(
    model, storage: NodeStorage, groups: list[str],
) -> list[Parameter]:
    """Create a list of parameters

        Parameters
        ----------
        model:
            Model of experiment
        storage: NodeStorage
            Storage of model where all necessary items are stored
        groups: list[str]
            List of parameters groups that will be used for Monte-Carlo method

        Returns
        -------
        list[Parameter]
            List of normalized parameters
    """
    parameters = []
    for group in groups:
        parameters.extend([
            parameter for parameter in walkvalues(storage[("parameters", "normalized") + properkey(model.systematic_uncertainties_groups()[group])])
        ])
    return parameters


def covariance_matrix_calculation(
    parameters: list[Parameter],
    generator: np.random.Generator,
    observation: Output,
    N: int,
    asimov: NDArray = None,
) -> NDArray:
    r"""Calculate absolute covariance matrix

    For the calculation used the next formula
    .. math:: cov_{ij} = \frac{1}{N}\sum_{k = 1}^{N}(x_i^k - \overline{x_i})(x_j^k - \overline{x_j}),

    where `x_i^k` is `i`-th bin value of `k`-th sample, `\overline{x_i}` is mean
    value of `i`-th bin, $N$ is normalization factor

    Here we are using simplified formula
    .. math:: cov_{ij} = \frac{1}{\text{norm}} \overline{x_i x_j} - \overline{x_i}\overline{x_j^A} - \overline{x_j}\overline{x_i^A} + \overline{x_i^A}\overline{x_i^A},

    where we averaging over all MC samples

    If Asimov observation is passed, the next formula is used
    .. math:: cov_{ij} = \frac{1}{\text{norm}} \overline{x_i x_j} - \overline{x_i}\overline{x_j^A} - \overline{x_j}\overline{x_i^A} + \overline{x_i^A}\overline{x_j^A},

    Parameters
    ----------
    parameters: list[Parameter]
        List of normalized parameters
    generator: np.random.Generator
        numpy generator of pseudo-random numbers
    observation: Output
        Observation of model that depends on parameters
    N: int
        Number of samples for calculation covariance matrices
    asimov: NDArray, optional
        Asimov observation (no fluctuation of parameters)

    Returns
    -------
    NDArray
        Two dimensional square array, absolute covariance matrix
    """
    observation_size = observation.data.shape[0]
    product_mean = np.zeros((observation_size, observation_size))
    observation_mean = np.zeros(observation_size)
    samples = np.zeros((N, observation_size))
    for i in range(N):
        variate_parameters(parameters, generator)
        observation_mean += observation.data
        product_mean += np.outer(observation.data, observation.data)
        samples[i] = observation.data
    observation_mean /= N
    product_mean /= N
    if asimov is not None:
        observation_mean_asimov = np.outer(observation_mean, asimov)
        observation_product_mean = observation_mean_asimov + observation_mean_asimov.T - np.outer(asimov, asimov)
        covariance_normalization_factor = N
    else:
        observation_product_mean = np.outer(observation_mean, observation_mean)
        covariance_normalization_factor = N - 1
    covariance_matrix_absolute = (product_mean - observation_product_mean) * N / covariance_normalization_factor
    return covariance_matrix_absolute


def covariance_matrix_calculation_alternative(
    parameters: list[Parameter],
    generator: np.random.Generator,
    observation: Output,
    N: int,
    asimov: NDArray = None,
) -> NDArray:
    r"""Calculate absolute matrix (alternative method)

    For the calculation used simplified formula
    .. math:: cov_{ij} = \frac{1}{\text{norm}} \overline{x_i x_j} - \overline{x_i}\overline{x_j^A} - \overline{x_j}\overline{x_i^A} + \overline{x_i^A}\overline{x_j^A},

    where `S` is `Nx(observation size)` matrix that contains all Monte-Carlo samples.

    Here we use a lot of memory to store every Monte-Carlo sample and we provide
    calculations via matrix product

    If Asimov observation is passed, the next formula is used
    .. math:: cov_{ij} = \frac{1}{\text{norm}} \overline{x_i x_j} - \overline{x_i}\overline{x_j^A} - \overline{x_j}\overline{x_i^A} + \overline{x_i^A}\overline{x_j^A},

    Parameters
    ----------
    parameters: list[Parameter]
        List of normalized parameters
    generator: np.random.Generator
        numpy generator of pseudo-random numbers
    observation: Output
        Observation of model that depends on parameters
    N: int
        Number of samples for calculation covariance matrices
    asimov: NDArray, optional
        Two dimensional square array, absolute covariance matrix

    Returns
    -------
    NDArray
        Absolute covariance matrix
    """
    observation_size = observation.data.shape[0]
    samples = np.zeros((N, observation_size))
    for i in range(N):
        variate_parameters(parameters, generator)
        samples[i] = observation.data
    if asimov is not None:
        samples_mean = asimov
        covariance_normalization_factor = N
    else:
        samples_mean = samples.mean(axis=0)
        covariance_normalization_factor = N - 1
    covariance_normalization_factor = N
    samples_diff = samples - samples_mean
    covariance_matrix_absolute = samples_diff.T @ samples_diff / covariance_normalization_factor
    return covariance_matrix_absolute


def calculate_correlation_matrix(covariance_matrix: NDArray) -> NDArray:
    r"""Calculate correlation matrix from covariance matrix

    $\mathrm{corr}_{ij} = \frac{\mathrm{cov}_{ij}}{\sqrt{\mathrm{cov}_{ii}\mathrm{cov}_{jj}}}$

    Parameters
    ----------
    covariance_matrix: NDArray
        covariance matrix

    Returns
    -------
    NDArray
        Correlation matrix
    """
    diagonal = np.diagonal(covariance_matrix)
    return covariance_matrix / np.outer(diagonal, diagonal)**0.5


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model = load_model(
        opts.version,
        model_options=opts.model_options,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    storage: NodeStorage = model.storage

    observation = storage["outputs.eventscount.final.concatenated.selected"]
    asimov = observation.data.copy()
    generator = np.random.Generator(np.random.MT19937(opts.seed))

    parameters = create_list_of_variation_parameters(
        model,
        storage,
        opts.systematic_parameters_groups,
    )

    if opts.alternative:
        covariance_absolute = covariance_matrix_calculation_alternative(
            parameters,
            generator,
            observation,
            opts.num,
            asimov if opts.asimov_as_mean else None,
        )
    else:
        covariance_absolute = covariance_matrix_calculation(
            parameters,
            generator,
            observation,
            opts.num,
            asimov if opts.asimov_as_mean else None,
        )

    covariance_relative = covariance_absolute / np.outer(asimov, asimov)
    correlation_matrix = calculate_correlation_matrix(covariance_absolute)

    cs = plt.matshow(covariance_absolute)
    plt.title("Covariance matrix (absolute)")
    plt.colorbar(cs)
    cs = plt.matshow(covariance_relative)
    plt.title("Covariance matrix (relative)")
    plt.colorbar(cs)
    cs = plt.matshow(correlation_matrix)
    plt.title("Correlation matrix")
    plt.colorbar(cs)

    plt.show()

    if opts.interactive:
        import IPython
        IPython.embed(colors="neutral")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )
    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="tsv",
        help="Data source type",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start IPython session",
    )

    plot = parser.add_argument_group("plot", "plotting related options")

    storage = parser.add_argument_group("storage", "storage related options")

    model = parser.add_argument_group("model", "model related options")
    model.add_argument(
        "--version",
        default="v0",
        choices=available_models(),
        help="model version",
    )
    model.add_argument(
        "--model-options", "--mo", default={}, help="Model options as yaml dict"
    )

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par", nargs=2, action="append", default=[], help="set parameter value"
    )

    cov = parser.add_argument_group("cov", "covariance parameters")
    pars.add_argument(
        "--systematic-parameters-groups", "--cov", default=[],
        nargs="+", help="Choose systematic parameters for building covariance matrix",
    )
    pars.add_argument(
        "--asimov-as-mean", action="store_true", help="Use Asimov data as mean",
    )
    pars.add_argument(
        "--alternative", action="store_true", help="Use alternative method for calculation (much memory-intensive)",
    )
    pars.add_argument(
        "--seed", default=0, help="Choose seed of randomization algorithm",
    )
    pars.add_argument(
        "-N", "--num", default=1000, help="Choose number of samples",
    )

    main(parser.parse_args())
