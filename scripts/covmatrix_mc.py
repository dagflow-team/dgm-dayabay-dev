#!/usr/bin/env python
import numpy as np
from argparse import Namespace
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING

from dagflow.logger import DEBUG as INFO4
from dagflow.logger import INFO1, INFO2, INFO3, set_level
from models import available_models, load_model

from dagflow.output import Output
from dagflow.storage import NodeStorage
from dagflow.parameters import Parameter
from numpy.typing import NDArray
if TYPE_CHECKING:
    from models import model_dayabay_v0

set_level(INFO1)


SYSTEMATIC_UNCERTAINTIES_GROUPS = {
    "oscprob": "oscprob",
    "eres": "detector.eres",
    "lsnl": "detector.lsnl_scale_a",
    "iav": "detector.iav_offdiag_scale_factor",
    "detector_relative": "detector.detector_relative",
    "energy_per_fission": "reactor.energy_per_fission",
    "nominal_thermal_power": "reactor.nominal_thermal_power",
    "snf": "reactor.snf_scale",
    "neq": "reactor.nonequilibrium_scale",
    "fission_fraction": "reactor.fission_fraction_scale",
    "bkg_rate": "bkg.rate",
    "hm_corr": "reactor_anue.spectrum_uncertainty.corr",
    "hm_uncorr": "reactor_anue.spectrum_uncertainty.uncorr",
}


def variate_parameters(parameters: list[Parameter], generator: np.random.Generator) -> None:
    """Randomize value of parameters via normal distribution N(0, 1)

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


def create_list_variation_parameters(
    storage: NodeStorage, groups: list[str],
) -> list[Parameter]:
    """Create list of parameters

        Parameters
        ----------
        storage: NodeStorage
            Storage of model where all necessary things are stored
        groups: list[str]
            List of parameters groups that will be used for Monte-Carlo method

        Returns
        -------
        list[Parameter]
            List of normalized parameters
    """
    parameters = []
    for group in groups:
        if group == "hm_corr":
            group_parameters = [
                storage.get_any(f"parameters.normalized.{SYSTEMATIC_UNCERTAINTIES_GROUPS[group]}")
            ]
        else:
            group_parameters = list(
                storage(f"parameters.normalized.{SYSTEMATIC_UNCERTAINTIES_GROUPS[group]}").walkvalues()
            )
        parameters.extend(group_parameters)
    return parameters


def covariance_matrix_calculation(
    parameters: list[Parameter],
    generator: np.random.Generator,
    observation: Output,
    N: int,
    observation_asimov: NDArray = None,
) -> tuple[NDArray, NDArray]:
    r"""Calculate absolute and relative covariance matrices

    For the calculation used simplified formula
    $cov_{ij} = \frac{1}{\text{norm}}\sum_{k = 1}^{N}(x_i^k - \bar{x_i})(x_j^k - \bar{x_j}),$
    where $x_i^k$ is `i`-th bin value of `k`-th sample,
    $\bar{x_i}$ is mean value of `i`-th bin,
    $\text{norm}$ is normalization factor.

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
    observation: NDArray, optional
        Asimov observation (no fluctuation of parameters)

    Returns
    -------
    tuple[NDArray, NDArray]
        tuple of NDArray, covariance absolute and covariance relative matrices
    """
    observation_size = observation.data.shape[0]
    product_mean = np.zeros((observation_size, observation_size))
    observation_mean = np.zeros(observation_size)
    samples = np.zeros((N, observation_size))
    for i in range(N):
        variate_parameters(parameters, generator)
        observation_mean += observation.data.copy()
        product_mean += np.outer(observation.data.copy(), observation.data.copy())
        samples[i] = observation.data.copy()
    observation_mean /= N
    product_mean /= N
    normalization_factor = N - 1
    if observation_asimov:
        observation_mean_asimov = np.outer(observation_mean, observation_asimov)
        observation_product_mean = observation_mean_asimov + observation_mean_asimov.T - np.outer(observation_asimov, observation_asimov)
        normalization_factor = N
    else:
        observation_product_mean = np.outer(observation_mean, observation_mean)
    covariance_matrix_absolute = (product_mean - observation_product_mean) * N / normalization_factor
    covariance_matrix_relative = covariance_matrix_absolute / observation_product_mean
    return covariance_matrix_absolute, covariance_matrix_relative


def covariance_matrix_calculation_alternative(
    parameters: list[Parameter],
    generator: np.random.Generator,
    observation: Output,
    N: int,
    observation_asimov: NDArray = None,
) -> tuple[NDArray, NDArray]:
    r"""Calculate absolute and relative covariance matrices

    For the calculation used simplified formula
    $cov_{ij} = \frac{1}{\text{norm}}\sum_{k = 1}^{N}(x_i^k - \bar{x_i})(x_j^k - \bar{x_j}),$
    where $x_i^k$ is `i`-th bin value of `k`-th sample,
    $\bar{x_i}$ is mean value of `i`-th bin,
    $\text{norm}$ is normalization factor.

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
    observation: NDArray, optional
        Asimov observation (no fluctuation of parameters)

    Returns
    -------
    tuple[NDArray, NDArray]
        tuple of NDArray, covariance absolute and covariance relative matrices
    """
    observation_size = observation.data.shape[0]
    samples = np.zeros((N, observation_size))
    for i in range(N):
        variate_parameters(parameters, generator)
        samples[i] = observation.data.copy()
    normalization_factor = N - 1
    if observation_asimov:
        samples_mean = observation_asimov
        normalization_factor = N
    else:
        samples_mean = samples.mean(axis=0)
    samples_diff = samples - samples_mean
    covariance_matrix_absolute = samples_diff.T @ samples_diff / normalization_factor
    covariance_matrix_relative = covariance_matrix_absolute / np.outer(samples_mean, samples_mean)
    return covariance_matrix_absolute, covariance_matrix_relative


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
    return covariance_matrix / np.outer(diagonal, diagonal)**.5


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    model: model_dayabay_v0 = load_model(
        opts.version,
        model_options=opts.model_options,
        source_type=opts.source_type,
        parameter_values=opts.par,
    )

    storage = model.storage

    observation = storage["outputs.eventscount.final.concatenated.selected"]
    observation_asimov = None
    if opts.asimov_as_mean:
        observation_asimov = observation.data.copy()
    generator = np.random.Generator(np.random.MT19937(opts.seed))

    parameters = create_list_variation_parameters(
        storage,
        opts.systematic_parameters_groups,
    )

    if opts.alternative:
        covariance_absolute, covariance_relative = covariance_matrix_calculation_alternative(
            parameters,
            generator,
            observation,
            opts.num,
            observation_asimov,
        )
    else:
        covariance_absolute, covariance_relative = covariance_matrix_calculation(
            parameters,
            generator,
            observation,
            opts.num,
            observation_asimov,
        )

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
        "--systematic-parameters-groups", "--cov",
        choices=SYSTEMATIC_UNCERTAINTIES_GROUPS.keys(), default=[],
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
