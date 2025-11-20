from pathlib import Path
import spikeinterface.extractors as si_extractors
import spikeinterface.widgets as si_widgets
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters
from probeinterface.plotting import plot_probe
import matplotlib.pyplot as plt

# NOTE! Use pip install kilosort==4.1.0, the newest kilosort
# release is not yet supported by spikeinterface

SHOW_PROBE = True  # definitely double check SI is loading the channel map correctly
SHOW_TRACES = True
RECORDING_PATH = Path(r"C:\Users\Jzimi\Desktop\rawdata\sub-1119617\ses-001\ephys\1119617_LSE1_shank12_g0")
OUTPUT_PATH = Path(r"C:\Users\Jzimi\Desktop\derivatives\1119617_LSE1_shank12_g0")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load and preprocess the data up to common median reference
# phase_shift corrects an offset introduced by NP hardware, then
# we have temporal filtering and common median reference.
# Plotting data at this stage gives an indication of data quality.
# SpikeInterface has other functions e.g. si_prepo.correct_motion
# which has better drift correction algorithms like DREDGE. But for
# now I'd start with kilosorts drift correction, I think KS4 outputs a drift
# map now which should give an indication of the drift in the recording.
# I think it can also be useful to run kilsort4 directly through their GUI
# and check running in kilosort4 vs. spikeinterface gives the same, or very similar, results.
# Also pay attention to the version of kilosort4 you are using, they are now making a lot
# of releases, but it's probably better to keep the same version through your entire experiment.
recording = si_extractors.read_spikeglx(
    RECORDING_PATH, stream_name="imec0.ap"
)

probe = recording.get_probe()

if SHOW_PROBE:
    plot_probe(probe)
    plt.show()


shift_rec = si_prepro.phase_shift(recording)
filter_rec = si_prepro.bandpass_filter(shift_rec, freq_min=300, freq_max=6000)

# After filtering, split the probe by shank and proceed
# for the rest of the preprocessing / sorting on the split shanks.
filt_rec_by_shank = filter_rec.split_by("group")

cmr_rec_by_shank = si_prepro.common_reference(filt_rec_by_shank, operator="median")

if SHOW_TRACES:
    # After the first stages of preprocessing, optionally
    # show the data per-shank.
    for shank_idx, rec in cmr_rec_by_shank.items():

        start_time = rec.get_times()[0]

        si_widgets.plot_traces(
            rec,
            time_range=(start_time, start_time + 0.5),
            order_channel_by_depth=True
        )
        plt.title(f"Shank index: {shank_idx}")
        plt.show()

# Run the sorter, old runs will be overwritten. It will first write the
# preprocessed recording to disk
si_sorters.run_sorter(
    recording=cmr_rec_by_shank,
    sorter_name="kilosort4",
    remove_existing_folder=True,
    folder=OUTPUT_PATH,
    # Turn off common average referencing, and set the filter
    # cutoff under what we used (you can't turn it off).
    # Because we already run these.
    do_CAR=False,
    highpass_cutoff=150,
)

