#!/usr/bin/env bash
AVG_DAYS=""
SEED=0

for chi2type in "full.chi2p_covmat_fixed" "full.chi2n_covmat" "full.chi2p_covmat_variable" "full.chi2cnp_covmat" "full.chi2p_iterative" "full.chi2cnp" "full.chi2p_unbiased"
do
    prefix="$(echo $chi2type | cut -d '.' -f 2)"
    OUTPUT_DIR="output/fit-model-b-mc/$SEED"
    mkdir -p ${OUTPUT_DIR}
    ./scripts/fit_dayabay_model.py  --version v0e \
        --mo "{dataset: b, future: [$AVG_DAYS]}" \
        --data-mc-mode poisson --seed $SEED \
        --source-type root \
        --chi2 "${chi2type}" --data data --use-free-spec \
        --output-plot-fit "${OUTPUT_DIR}/${prefix}-fit_2d.pdf" \
        --output-fit "${OUTPUT_DIR}/${prefix}.yaml" \
        --output-plot-spectra "${OUTPUT_DIR}/${prefix}_{}.pdf" > "${OUTPUT_DIR}/${prefix}.log" &
    sleep 20
done

