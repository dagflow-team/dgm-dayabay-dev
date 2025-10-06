# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
