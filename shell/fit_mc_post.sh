#!/usr/bin/env bash

cd output/fit-model-b-mc/

for dirname in $(ls -d *)
do
    cd $dirname
    echo $PWD $dirname
    for chi2type in "chi2p_covmat_fixed" "chi2n_covmat" "chi2p_covmat_variable" "chi2cnp_covmat" "chi2p_iterative" "chi2cnp" "chi2p_unbiased"
    do
        pdftk "${chi2type}-fit_2d.pdf" ${chi2type}_*.pdf cat output "../${dirname}-${chi2type}.pdf"
    done
    cd ../
    python ../../fit2-data-to-table.py --filename "${dirname}/*.yaml" \
        --dataset-file ../../../spectra-comparison/sysu/oscillation_parameters2.yaml > "${dirname}-table.html"
done

