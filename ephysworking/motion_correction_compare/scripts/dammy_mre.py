"""
This is a quick script to visualise the output of bandpass filter
and CMR on the recordings. Output will either save a list of
images to the session folder (will be overwritten each
time the script is run). Otherwise it will display in the
current GUI.

Pictures are output to
/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/deriatives/...
"""

from pathlib import Path
import platform
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.widgets as si_widgets
from spikeinterface.sorters import Kilosort2_5Sorter
import probeinterface
from ephysworking.motion_correction_compare.utils import gen_probe_group
from ephysworking.utils import plot_list_of_recordings

from probeinterface.plotting import plot_probe, plot_probe_group

if platform.system() == "Windows":
    base_path = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\dammy")
else:
    base_path = Path(
        r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy"
    )


# Set subject / session information
sub = "DO79"
ses = "240404_001"   # e.g. "240404_001" # "240109_001"
probe_idx = 0        # 0 or 1
shank_idx = 2        # (0-3 probe 1, 4-7 probe 2)
save_plots = False   # save to folder is True, otherwise display now with matplotlib.
show_probe_plot = False
plot_mode = "line"    # "map" or "line"

get_ses_path = lambda toplevel: base_path / toplevel / sub / ses


# Load the raw data
recording_path = list(get_ses_path("rawdata").glob("**/Record Node 101/experiment*/recording*"))

assert len(recording_path) == 1, f"{sub} {ses} has unexpected number of recordings."

raw_rec_noprobe = si_extractors.read_openephys(recording_path[0].as_posix())

probes = gen_probe_group()  # vendored from Dammy's code

if show_probe_plot and not save_plots:
    plot_probe_group(probes, with_contact_id=True)  # with_contact_id with_device_index
    plt.show()


# Split the two probes
raw_rec_two_probe = raw_rec_noprobe.set_probegroup(probes)
raw_rec_one_probe = raw_rec_two_probe.split_by("group")[probe_idx]


# Split by shank, take one arbitrarily
raw_rec_one_probe.set_property(
    "group", np.int32(raw_rec_one_probe.get_probe().shank_ids)
)
raw_rec = raw_rec_one_probe.split_by("group")[shank_idx]


# Preprocess in SI
filtered_rec = si_preprocessing.bandpass_filter(raw_rec, freq_min=300, freq_max=6000)
cmr_rec = si_preprocessing.common_reference(filtered_rec, operator="median")


# Plot all outputs for visual comparison.
if save_plots:
    output_path = get_ses_path("derivatives") / ses / "images"
    if output_path.is_dir():
        output_path.rmdir()

    plot_list_of_recordings(
        get_ses_path("derivatives"),
        ses,
        raw_rec.get_times(),
        ["cmr"],
        [cmr_rec],
        mode=plot_mode,
    )
else:
    start_time = 100
    stop_time = 100.2

    si_widgets.plot_traces(
        cmr_rec,
        order_channel_by_depth=True,
        time_range=(start_time, stop_time),
        mode=plot_mode,
        show_channel_ids=True,
    )
    plt.show()
