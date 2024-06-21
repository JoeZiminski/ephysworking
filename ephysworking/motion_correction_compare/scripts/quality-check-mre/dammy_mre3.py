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
import spikeinterface.preprocessing as si
from copy import copy
import os

def preprocess(recording,filter_range=(300,9000),rec_dir=''):
    bad_channels_ids, _ = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channels_ids)
    recording = si.bandpass_filter(recording, freq_min=filter_range[0], freq_max=filter_range[1])
    recording = cmr_by_shank(recording)
    job_kwargs = dict(n_jobs=os.cpu_count(), chunk_duration='1s', progress_bar=True)
    # recording, motion_info = correct_drift(recording,'nonrigid_fast_and_accurate',rec_dir,job_kwargs)
    # recording, motion_info = correct_drift(recording,'nonrigid_accurate',rec_dir,job_kwargs)
    return recording


def cmr_by_shank(recording):
    probe_df = recording.get_probe().to_dataframe(complete=True)
    main_ids = copy(recording._main_ids)
    recording._main_ids = recording.ids_to_indices()
    # chan_letters = np.unique(
    #     [chan_id.split('_')[0] if len(chan_id.split('_')) > 1 else [''] for chan_id in recording._main_ids])

    cmr_group_param = 'shank_ids'
    cmr_groups_idx = [probe_df[probe_df[cmr_group_param] == i]['device_channel_indices'].astype(int).to_list()
                      for i in probe_df[cmr_group_param].unique()]
    # if chan_letters[0]:
    #     cmr_groups = [[f'{chan_letters[0]}_CH{int(cid) + 1}' if int(
    #         cid) < 64 else f'{chan_letters[1]}_CH{int((int(cid) + 1) / 2)}' for cid in group] for group in
    #                   cmr_groups_idx]
    # else:
    #     cmr_groups = [[f'CH{int(cid) + 1}' for cid in group] for group in cmr_groups_idx]
    # recording_cmr = si.common_reference(recording, reference='global', operator='median', groups=cmr_groups)
    recording_cmr = si.common_reference(recording, reference='global', operator='median', groups=cmr_groups_idx)
    recording_cmr._main_ids = main_ids
    return recording_cmr

path_ = Path(r"Y:\Dammy\ephys\DO82_2024-04-16_001\Record Node 101\experiment1\recording1")

raw_rec_noprobe = si_extractors.read_openephys(path_)

probes = gen_probe_group()  # vendored from Dammy's code

# Split the two probes
raw_rec_two_probe = raw_rec_noprobe.set_probegroup(probes)

probe_idx = 0
shank_idx = 0


# Split the two probes
raw_rec_one_probe = raw_rec_two_probe.split_by("group")[probe_idx]
preprocessed = si_preprocessing.bandpass_filter(raw_rec_one_probe, freq_min=600, freq_max=6000)


# preprocessed = preprocess(raw_rec_one_probe, filter_range=(300, 9000))

# Split by shank, take one arbitrarily
preprocessed.set_property(
    "group", np.int32(preprocessed.get_probe().shank_ids)
)
preprocessed = preprocessed.split_by("group")[shank_idx]

start_time = 1500
stop_time =1501

si_widgets.plot_traces(
    preprocessed,
    order_channel_by_depth=True,
    time_range=(start_time, stop_time),
    mode="line",
    show_channel_ids=True,
    return_scaled=True
)

plt.show()

