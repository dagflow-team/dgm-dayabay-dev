from dagflow.bundles.load_parameters import load_parameters
from dictwrapper.dictwrapper import DictWrapper
from pathlib import Path

from pprint import pprint

def model_dayabay_v0():
	storage = DictWrapper({}, sep='.')
	datasource = Path('data/dayabay-v0')
	storage |= load_parameters({'path': 'ibd', 'load': datasource/'parameters/pdg2012.yaml'})
	storage |= load_parameters({'path': 'reactor', 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'})
	storage |= load_parameters({'path': 'reactor', 'load': datasource/'parameters/detector_eres.yaml'})

	pprint(storage.object)
