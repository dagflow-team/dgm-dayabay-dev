#!/usr/bin/env fish

set ver v0b
set jobs 6

for mode in 250keV 50keV_nonscaled 50keV_scaled 50keV_scaled_approx
    sem -j $jobs \
    ./scripts/covmatrix_dayabay_v0.py --version "$ver" output/covariance_matrix_"$ver"_"$mode".hdf5 \
                                      --mo "'{anue_spectrum_model: $mode}'" \> output/covariance_matrix_"$ver"_"$mode".out
end

sem --wait
