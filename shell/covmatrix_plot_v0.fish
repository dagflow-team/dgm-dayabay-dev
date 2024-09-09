#!/usr/bin/env fish

set jobs 6

# for file in output/covariance_matrix_v0b_*.hdf5
#     sem -j $jobs ./scripts/covmatrix_plot.py $file -o
# end

for file in \
    output/covariance_matrix_v0b_50keV_nonscaled.hdf5 \
    output/covariance_matrix_v0b_50keV_scaled.hdf5 \
    output/covariance_matrix_v0b_50keV_scaled_approx.hdf5
    set -l dirname (dirname $file)
    set -l basename (basename $file .hdf5)
    sem -j $jobs ./scripts/covmatrix_plot.py $file \
                                --subtract output/covariance_matrix_v0b_250keV.hdf5 \
                                -o $dirname/"$basename"_diff.pdf
end

sem -j $jobs ./scripts/covmatrix_plot.py output/covariance_matrix_v0b_ffnorm.hdf5 \
                            --subtract output/covariance_matrix_v0b_50keV_scaled_approx.hdf5 \
                            -o output/covariance_matrix_v0b_ffnorm_diff.pdf


sem --wait
