from dagflow.bundles.load_parameters import load_parameters
from pathlib import Path

from pprint import pprint

def model_dayabay_v0():
	datasource = Path('data/dayabay-v0')
	vars1 = load_parameters({'path': 'ibd', 'load': datasource/'parameters/pdg2012.yaml'})
	vars2 = load_parameters({'path': 'reactor', 'load': datasource/'parameters/reactor_e_per_fission.yaml'})

	pprint(vars1.object)
	pprint(vars2.object)
