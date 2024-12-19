#!/usr/bin/env fish

set modes $argv

./scripts/cmp_models_v0.py v0c v0d --mo-b "{future: [data-a], source_type: hdf5}" --hist outputs.eventscount.fine.bkg -s \
    --norm-par-a bkg.rate.acc.6AD.AD11 2 \
    --norm-par-b bkg.rate_scale.acc.6AD.AD11 2
