from pathlib import Path
import platform
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.widgets as si_widgets
from spikeinterface.sorters import Kilosort2_5Sorter

from ephysworking.motion_correction_compare.utils import gen_probe_group
from ephysworking.utils import plot_list_of_recordings


if platform.system() == "Windows":
    base_path = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\dammy")
    ks_path = Path(
        r"X:\neuroinformatics\scratch\jziminski\git-repos\forks\Kilosort_2-5_nowhiten"
    )
else:
    base_path = Path(
        r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy"
    )
    ks_path = Path(
        "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/git-repos/forks/Kilosort_2-5_nowhiten"
    )

Kilosort2_5Sorter.set_kilosort2_5_path(ks_path.as_posix())

# Set subject / session information
sub = "DO79"
ses = "240109_001"  # "240404_001" # "240109_001"
probe_idx = 0
shank_idx = 1
save_plots = True

get_ses_path = lambda toplevel: base_path / toplevel / sub / ses


# Load the raw data
recording_path = list(get_ses_path("rawdata").glob("**/Record Node 101*"))

assert len(recording_path) == 1, f"{sub} {ses} has unexpected number of recordings."

raw_rec_noprobe = si_extractors.read_openephys(recording_path[0].as_posix())
probes = gen_probe_group()


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
    plot_list_of_recordings(
        get_ses_path("derivatives"),
        ses,
        raw_rec.get_times(),
        ["cmr"],
        [cmr_rec],
    )
else:
    start_time = 1
    stop_time = 2
    si_widgets.plot_traces(
        cmr_rec,
        order_channel_by_depth=False,
        time_range=(start_time, stop_time),
        mode="map",
        show_channel_ids=True,
    )
    plt.show()
