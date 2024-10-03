#!/usr/bin/env fish

set modes $argv

set modes_simple pdg xsec conversion hm-spectra lsnl-curves short-baselines
set keys_other lsnl-matrix fix-neq-shape
set keys_all $modes_simple $keys_other
set keys_all_str (string join ", " -- $keys_all)
set modes_other lsnl-matrix0 $keys_other
set modes_all $modes_other $modes_simple
if test ! $argv
    echo Modes: $modes_all all
end

for opt in pdg $modes_simple
    contains $opt -- $modes; or continue

    ./scripts/cmp_models_v0.py v0b v0 --mo-a "{future: $opt}" -s
end

contains lsnl-matrix0 -- $modes; and \
./scripts/cmp_models_v0.py v0b v0 --par detector.lsnl_scale_a.pull0 1 -s

contains lsnl-matrix -- $modes; and \
    ipython --pdb -- \
./scripts/cmp_models_v0.py v0b v0 --mo-a "{future: lsnl-matrix}" --par detector.lsnl_scale_a.pull0 1 --ylim -0.1 0.1 -s

contains all -- $modes; and \
./scripts/cmp_models_v0.py v0b v0 --mo-a "{future: [$keys_all_str]}" --par detector.lsnl_scale_a.pull0 1 -s

