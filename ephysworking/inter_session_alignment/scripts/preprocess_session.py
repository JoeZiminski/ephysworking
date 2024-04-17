# Dependencies: spikeinterface==0.98.2, probeinterface, matplotlib, scipy

import probeinterface.plotting
from spikeinterface import extract_waveforms
from spikeinterface.extractors import read_spikeglx
from spikeinterface.preprocessing import phase_shift, bandpass_filter, common_reference, correct_motion
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sorters import run_sorter
from pathlib import Path
from probeinterface.plotting import plot_probe, plot_probe_group
import matplotlib.pyplot as plt
from spikeinterface import curation
from spikeinterface.widgets import plot_timeseries
import spikeinterface as si  # TODO
import argparse
import numpy as np
from calculate_histogram_shift import save_plot
import histogram_generation


parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--sub")
parser.add_argument("--ses")

opts = parser.parse_args()

project_path = Path(opts.path)
sub = opts.sub
ses = opts.ses

data_path = project_path / "rawdata" / sub
output_path = project_path / "derivatives" / sub / ses / "preprocessing"

job_kwargs = dict(chunk_duration="1s", n_jobs=20, progress_bar=True)

assert (data_path / ses).is_dir(), f"{data_path / ses} could not be found."

raw_recording = read_spikeglx(data_path / ses)

# Run the preprocessing steps
shifted_recording = phase_shift(raw_recording)
filtered_recording = bandpass_filter(shifted_recording, freq_min=300, freq_max=6000)
referenced_recording = common_reference(
    filtered_recording, reference="global", operator="median"
)
preprocessed_recording, motion_info = correct_motion(recording=referenced_recording, preset="kilosort_like", output_motion_info=True, estimate_motion_kwargs={"rigid": True}, **job_kwargs)

output_path.mkdir(exist_ok=True, parents=True)

save_plot(
    motion_info["temporal_bins"],
    motion_info["motion"],
    "Time (s)",
    "Displacement (μm)",
    output_path / "motion_per_channel_before_correction.png"
)

average_motion_over_channels = np.mean(motion_info["motion"], axis=1)

save_plot(
    motion_info["temporal_bins"],
    average_motion_over_channels,
    "Time (s)",
    "Displacement (μm)",
    output_path / "avg_motion_before_correction.png"
)

peaks, peak_locations = histogram_generation.get_peaks_and_peak_locations(preprocessed_recording, job_kwargs)
corrected_motion, temporal_bins, _ = estimate_motion(preprocessed_recording, peaks, peak_locations, method="iterative_template", rigid=True)

# TODO: own function
save_plot(
    temporal_bins,
    corrected_motion,
    "Time (s)",
    "Displacement (μm)",
    output_path / "motion_per_channel_after_correction.png"
)

average_motion_over_channels = np.mean(corrected_motion, axis=1)

save_plot(
    temporal_bins,
    average_motion_over_channels,
    "Time (s)",
    "Displacement (μm)",
    output_path / "avg_motion_after_correction.png"
)


preprocessed_recording.save(folder=output_path / "si_recording")
