#!/usr/bin/env fish

set modes $argv

set modes_simple pdg xsec conversion hm-spectra lsnl-curves
if test ! $argv
    echo Modes: lsnl-matrix0 lsnl-matrix fix-neq-shape $modes_simple
end

for opt in pdg $modes_simple
    contains $opt -- $modes; and \
    ./scripts/cmp_models_v0.py v0b v0 --mo-a "{future: $opt}" -s
end

contains lsnl-matrix0 -- $modes; and \
./scripts/cmp_models_v0.py v0b v0 --par detector.lsnl_scale_a.pull0 1 -s

contains lsnl-matrix -- $modes; and \
    ipython --pdb -- \
./scripts/cmp_models_v0.py v0b v0 --mo-a "{future: lsnl-matrix}" --par detector.lsnl_scale_a.pull0 1 --ylim -0.1 0.1 -s

# TODO: fix-neq-shape
