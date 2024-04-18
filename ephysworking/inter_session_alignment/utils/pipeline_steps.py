"""

"""
from pathlib import Path

import histogram_generation
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si
from spikeinterface import widgets
from spikeinterface.sortingcomponents.motion_interpolation import (
    interpolate_motion,
)


def save_motion_histograms(sub_path, ses):
    """ """
    ses_path = sub_path / ses
    recording_path = ses_path / "preprocessing" / "si_recording"
    recording = si.load_extractor(recording_path)

    recording_npy_path = ses_path / "motion_npy_files"
    recording_npy_path.mkdir(exist_ok=True, parents=True)

    # Make and save activity histograms
    peaks, peak_locations, histogram, temporal_bins, spatial_bins = (
        histogram_generation.make_single_motion_histogram(
            recording,
        )
    )

    np.save(recording_npy_path / "peaks.npy", peaks)
    np.save(recording_npy_path / "peak_locations.npy", peak_locations)
    np.save(recording_npy_path / "histogram.npy", histogram)
    np.save(recording_npy_path / "temporal_bins.npy", temporal_bins)
    np.save(recording_npy_path / "spatial_bins.npy", spatial_bins)

    # Save plot of histograms
    fig, ax = plt.subplots()
    ax.plot(spatial_bins[:-1], histogram)
    ax.set_xlabel("Contact Position (μm)")
    ax.set_ylabel("Frequency")
    ax.figure.legend([ses])
    fig.savefig(ses_path / "motion_histogram.png")
    plt.close(fig)


def align_to_first_session(sub_path, first_session, second_session):
    """ """
    # Set paths.
    pp_one_path = sub_path / first_session
    pp_two_path = sub_path / second_session

    pp_one_npy_path = Path(pp_one_path) / "motion_npy_files"
    pp_two_npy_path = Path(pp_two_path) / "motion_npy_files"

    # Load histogram information.
    pp_one_rec = si.load_extractor(pp_one_path / "preprocessing" / "si_recording")
    pp_two_rec = si.load_extractor(pp_two_path / "preprocessing" / "si_recording")

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
    # TODO: do not change the foldernames without changing in 'save_data_plots'
    # put in the same place.
    shifted_output_path = Path(pp_two_path) / "shifted_data" / "si_recording"

    pp_two_motion = np.empty((pp_two_temporal_bins.size, pp_two_spatial_bins.size))
    pp_two_motion[:, :] = scaled_shift

    # TODO: double check sign convention
    # border_mode can be: remove_channels  force_zeros
    shifted_pp_two_rec = interpolate_motion(
        recording=pp_two_rec,
        motion=-pp_two_motion,
        temporal_bins=pp_two_temporal_bins,
        spatial_bins=pp_two_spatial_bins,
        border_mode="force_zeros",
        spatial_interpolation_method="kriging",
        sigma_um=30.0,
    )
    shifted_pp_two_rec.save(folder=shifted_output_path)

    # Make motion histogram on the shifted data. TODO: this is not saved.
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
    ax.set_xlabel("Contact Position (μm)")
    ax.set_ylabel("Frequency")
    ax.figure.legend([first_session, second_session, f"Shifted: {second_session}"])
    fig.savefig(Path(pp_two_path) / "shifted_motion_histogram.png")
    plt.close(fig)


# TODO: this is almost exactly the same as ephysdata.utils.plot_list_of_recordings()
def save_data_plots(sub_path, ses):  # TODO: MOVE TO UTILS

    ses_path = sub_path / ses

    for rec_type in ["preprocessing", "shifted_data"]:

        recording_path = ses_path / rec_type / "si_recording"
        if not recording_path.is_dir():
            continue

        recording = si.load_extractor(recording_path)

        for shank_id, shank_rec in recording.split_by("group").items():

            # assert more than one segment

            # plot shanks separately
            all_times = shank_rec.get_times(segment_index=0)

            # not quite the end so bin doesn't go over edge
            # TODO: assumes recording is at least 100 seconds long.
            quarter_times = np.quantile(all_times, (0, 0.25, 0.5, 0.75, 0.95))
            start_times = [
                np.random.uniform(quarter_times[i], quarter_times[i + 1])
                for i in range(len(quarter_times) - 1)
            ]

            bin_sizes = (0.05, 1, 5)

            for start in start_times:
                for bin in bin_sizes:
                    fig, ax = plt.subplots()

                    widgets.plot_traces(
                        shank_rec,
                        order_channel_by_depth=True,
                        time_range=(start, start + bin),
                        ax=ax,
                    )
                    format_start = f"{start:0.2f}"
                    ax.set_title(f"{ses}\n start time: {format_start}, bin size: {bin}")
                    fig.savefig(
                        recording_path.parent
                        / f"shank-{shank_id}_start-{format_start}_bin-{bin}.png"
                    )
                    plt.close(fig)
