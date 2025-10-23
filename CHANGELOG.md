# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.5.0] - 2025-10-xx

- feature: add `v1a_distorted` to study extreme spectral distortions.
- feature: add `covariance_groups` parameter to control passed nuisance parameters to covarince matrix, works only with `strict=False`. Supported for `v1a` and `v1a_distorted`.
- feature: add `pull_groups` parameter to control passed nuisance parameters to `nuisance.extra_pull`, works only with `strict=False`. Supported for `v1a` and `v1a_distorted`.

## [0.4.2] - 2025-10-17

- chore: disable `numba` caching as it may cause problems for parallel execution. Configurable.
- chore: step `dgm-reactor-neutrino` dependence version to 0.2.2.

## [0.4.1] - 2025-10-16

- feature: validate the version from `data_information.yaml` and use it to determine the
  `source_type` (data format).
- feature: check overridden indices exist (model v1a).
- fix: apply `Abs` transformation to scaled fission fractions. Fit becomes much stable.
- fix: update keys related to graph visualization for [scripts/scripts/run_dayabay.py](extras/scripts/run_dayabay.py).

## [0.4.0] - 2025-10-07

- feature: configure the final binning (v1a) via configuration file of an argument.

## [0.3.0] - 2025-10-06

- feature: model `v1a`, add `detector_included` index to select detectors for the χ² construction.
- update: auto detection of source type skip `parameters` directory natively. Docstring was added.

## [0.2.3] - 2025-10-02

- feature: Add `switch_data` method from version v1a. It can switch model output between real data and Asimov.
- feature: Support all the inputs via `uproot` (`ROOT` is supported just as before).
- feature: v1a, read antineutrino spectra from tsv file (not python).

## [0.2.2] - 2025-09-25

- feature: Add key `leading_mass_splitting_3l_name` to switch between |Δm²₃₂| and |Δm²₃₁| to v1a version. In Day Bay official analysis |Δm²₃₂| as leading mass splitting.
- feature: Add key `override_cfg_files` that allows to override paths to configuration files. Key is supported for models v1+.
- feature: Remove `source_type` parameter from version v1+. Now `source_type` is determined automatically from `path_data`. To override default value, just path `path_data` to your model.

## [0.2.1] - 2025-09-08

- chore: Unify and version models and scripts, sync models with public (non-dev).
- fix: Unify the code, which disables implicit numpy multithreading.
- chore: Proper verbosity configuration via `set_verbosity`.
- chore: update `dag-modelling` dependency version.

## [0.2] - 2025-07-29

- First PYPI version.

## [0.1] - 2025-07-29

The first (pre)release version.
