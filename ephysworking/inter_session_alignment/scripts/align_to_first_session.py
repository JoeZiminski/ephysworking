from pathlib import Path

import histogram_generation
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si
from spikeinterface.sortingcomponents.motion_interpolation import (
    interpolate_motion,
)

# TODO: need to do a line-by-line check of entire script!!
# TODO: motion correction only supported for single-segment recordings

# TODO: need to check these carefully, how are they used during the interpolation?
#  spatial_bins = np.array([np.min(y_pos), np,max(y_pos)]) # TODO: this operation is done above
# TODO: need to add nonlinear drift. Need to try and implemnent propery nonlinear transformation.
# 1) get temporal / spatial bins
# 2) Actually, split this script into create / save the histogram + a shift to shift data
# 3) Save output as PNG. Save some actual traces?


if __name__ == "__main__":

    PLATFORM = "l"  # "w" or "l"
    if PLATFORM == "w":
        base_path = Path(r"X:\neuroinformatics\scratch\jziminski\1119617\derivatives")
    elif PLATFORM == "l":
        base_path = Path(
            "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/1119617/derivatives"
        )

    first_session = "1119617_LSE1_shank12_g0"
    second_session = "1119617_pretest1_shank12_g0"

    pp_one_path = base_path / first_session
    pp_two_path = base_path / second_session

    pp_one_rec = si.load_extractor(pp_one_path / "preprocessing" / "si_recording")
    pp_two_rec = si.load_extractor(pp_two_path / "preprocessing" / "si_recording")

    pp_one_npy_path = Path(pp_one_path) / "motion_npy_files"
    pp_two_npy_path = Path(pp_two_path) / "motion_npy_files"

    pp_one_histogram = np.load(pp_one_npy_path / "histogram.npy")
    pp_one_spatial_bins = np.load(pp_one_npy_path / "spatial_bins.npy")

    pp_two_histogram = np.load(pp_two_npy_path / "histogram.npy")
    pp_two_temporal_bins = np.load(pp_two_npy_path / "temporal_bins.npy")
    pp_two_spatial_bins = np.load(pp_two_npy_path / "spatial_bins.npy")

    assert np.array_equal(pp_one_spatial_bins, pp_two_spatial_bins)

    # Calculate shift from pp_two to pp_one
    scaled_shift, y_pos = histogram_generation.calculate_scaled_histogram_shift(
        pp_one_rec, pp_two_rec, pp_one_histogram, pp_two_histogram
    )

    # Shift the second session
    shifted_output_path = Path(pp_two_path) / "shifted_data" / "si_recording"

    pp_two_motion = np.empty((pp_two_temporal_bins.size, pp_two_spatial_bins.size))
    pp_two_motion[:, :] = scaled_shift

    shifted_pp_two_rec = interpolate_motion(
        recording=pp_two_rec,
        motion=-pp_two_motion,  # TODO: double check sign convention
        temporal_bins=pp_two_temporal_bins,
        spatial_bins=pp_two_spatial_bins,
        border_mode="force_zeros",  # remove_channels  force_zeros
        spatial_interpolation_method="kriging",
        sigma_um=30.0,
    )
    shifted_pp_two_rec.save(folder=shifted_output_path)

    #  TODO: use the function from other script, move saving apaprantus into the function and allow switch.
    (
        shifted_pp_two_peaks,
        shifted_pp_two_peak_locations,
        shifted_pp_two_histogram,
        _,
        shifted_pp_two_spatial_bins,
    ) = histogram_generation.make_single_motion_histogram(
        shifted_pp_two_rec,
    )

    # Save plot of histograms
    fig, ax = plt.subplots()
    ax.plot(pp_one_spatial_bins[:-1], pp_one_histogram)
    ax.plot(pp_two_spatial_bins[:-1], pp_two_histogram)
    ax.plot(shifted_pp_two_spatial_bins[:-1], shifted_pp_two_histogram)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("shift (um I think)")
    ax.figure.legend([first_session, second_session, f"Shifted: {second_session}"])
    fig.savefig(shifted_output_path.parent / "shifted_motion_histogram.png")
    plt.close(fig)
