[![python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)
[![pipeline](https://git.jinr.ru/dag-computing/dayabay-model/badges/main/pipeline.svg)](https://git.jinr.ru/dag-computing/dayabay-model/commits/main)
[![coverage report](https://git.jinr.ru/dag-computing/dayabay-model/badges/main/coverage.svg)](https://git.jinr.ru/dag-computing/dayabay-model/-/commits/main)
<!--- Uncomment here after adding docs!
[![pages](https://img.shields.io/badge/pages-link-white.svg)](http://dag-computing.pages.jinr.ru/dayabay-model)
-->

# Dayabay Model

The model of the Daya Bay experiment.

## Data

The data is located in a [restricted repository](https://git.jinr.ru/dag-computing/dayabay-data-all), to enable it use:

```sh
git lfs install --skip-repo # initialize the LFS (if needed)
git clone git@git.jinr.ru:dag-computing/dayabay-data-all.git data
```

It should be located in the `data/` folder. Note that git LFS is used to store the data files. It should be initialized (once) if not used before.

## Quick start

The repository depends on a few submodules. To initialize them use:

```sh
git submodule sync
git submodule update --init --recursive -- submodules/dag-flow
git submodule update --init -- submodules/dagflow-reactornueosc submodules/dagflow-detector submodules/dagflow-statistics
python3 -m pip install -r requirements.txt
```

this will:
1. Synchronize submodules
2. Check out the core submodule `dag-flow` recursively
3. Check out the physics repositories
