from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import calculate_histogram_shift
import numpy as np
from spikeinterface.sortingcomponents import motion_estimation
from spikeinterface.sortingcomponents.peak_localization import localize_peaks


def make_single_motion_histogram_per_session(recording, peaks, peak_locations):
    """"""
    spatial_bins = motion_estimation.get_spatial_bin_edges(recording, direction="y", margin_um=0.0, bin_um=10.0)  # TODO: this defines the histogram binsize

    # make a 3D histogram
    motion_histograms, temporal_hist_bin_edges, spatial_hist_bin_edges = \
        motion_estimation.make_2d_motion_histogram(
        recording,
        peaks,
        peak_locations,
        direction="y",
        bin_duration_s=recording.get_times()[-1] + 0.1, # bin_duration_s,  # TODO: could make much larger, will this crash on super long files?
        spatial_bin_edges=spatial_bins,
    )

    assert motion_histograms.shape[0] == 1
    return motion_histograms[0, :], temporal_hist_bin_edges, spatial_hist_bin_edges


def make_single_motion_histogram(recording):
    """"""
    peaks, peak_locations = get_peaks_and_peak_locations(recording)
    histogram, temporal_hist_bin_edges, spatial_hist_bin_edges = make_single_motion_histogram_per_session(recording, peaks, peak_locations)

    return peaks, peak_locations, histogram, temporal_hist_bin_edges, spatial_hist_bin_edges

# TODO: this does not handle the same defaults as spikeinterface defaults
# this is super confusing
# https://github.com/SpikeInterface/spikeinterface/blob/6d382a2906dfc44bf060af5ef7caf83e475be9f3/src/spikeinterface/preprocessing/motion.py#L217
def get_peaks_and_peak_locations(recording, job_kwargs=None):
    if not job_kwargs:
        job_kwargs = {}
    peaks = detect_peaks(recording=recording, method="locally_exclusive", detect_threshold=8.0, **job_kwargs)
    peak_locations = localize_peaks(recording=recording, peaks=peaks, method="center_of_mass", radius_um=75.0, **job_kwargs)
    return peaks, peak_locations


def calculate_scaled_histogram_shift(recording_1, recording_2, histogram_1, histogram_2):
    """"""
    assert histogram_1.size == histogram_2.size

    # Calculate the shift and interpolate
    pos_1, y_um_1 = get_probe_info(recording_1, histogram_1.size)
    pos_2, y_um_2 = get_probe_info(recording_2, histogram_2.size)

    assert np.array_equal(pos_1, pos_2)
    assert y_um_1 == y_um_2

    shift_2_to_1 = calculate_histogram_shift.calculate_shift(histogram_1, histogram_2)

    # TODO: need to adjust this scale if changing num bins
    #  in the histogram!!!?? NOT ANYMORE DONT THINK
    scaled_shift = shift_2_to_1 * y_um_2  # TODO: is this um?

    return scaled_shift, pos_1


def get_probe_info(recording, histogram_n):  # TODO: get info
    probe = recording.get_probe()
    dim = ["x", "y", "z"].index("y")
    pos = probe.contact_positions[:, dim]
    y_um = (np.max(pos) - np.min(pos)) / histogram_n
    return pos, y_um
