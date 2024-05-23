"""
This is a quick script to visualise the output of bandpass filter
and CMR on the recordings. Output will either save a list of
images to the session folder (will be overwritten each
time the script is run). Otherwise it will display in the
current GUI.

Pictures are output to
/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy/deriatives/...
"""


import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.widgets as si_widgets
from probeinterface.plotting import plot_probe_group

from ephysworking.motion_correction_compare.motion_utils import gen_probe_group
from ephysworking.utils import plot_list_of_recordings
import spikeinterface as si

path_ = Path(r"Y:\Dammy\ephys\DO82_240416_concat\sorting_no_si_drift\preprocessed")
output_path = path_ / "images"

recording = si.load_extractor(path_)
# [Probe - 59ch - 4shanks, Probe - 39ch - 4shanks] (is 59 ch and 39 ch okay?)

probe_idx = 0
shank_idx = 0

recording_by_probe = recording.split_by("group")[probe_idx]
print(recording_by_probe.get_probe())
# Probe - 59ch - 4shanks
# yes thats fine, bad channes removed

# I think this also contains the same info as the dataframe
recording_by_probe.set_property(
    "group", np.int32(recording_by_probe.get_probe().shank_ids)
)
recording_by_probe_shank = recording_by_probe.split_by("group")[shank_idx]

save_plots = False
plot_mode = "line"

start_time = 300
stop_time =301

si_widgets.plot_traces(
    recording_by_probe_shank,
    order_channel_by_depth=True,
    time_range=(start_time, stop_time),
    mode=plot_mode,
    show_channel_ids=True,
    return_scaled=True
)
plt.show()

