import platform
import runpy
from pathlib import Path

import numpy as np
from spikeinterface import extractors as si_extractors
from spikeinterface import load_extractor
from spikeinterface import preprocessing as si_preprocessing
from spikeinterface import sorters as si_sorters
from spikeinterface.core import BinaryRecordingExtractor

from ephysworking.motion_correction_compare.utils import gen_probe_group
from ephysworking.utils import full_plot_motion, plot_list_of_recordings


# Setup
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


from spikeinterface.sorters import Kilosort2_5Sorter

Kilosort2_5Sorter.set_kilosort2_5_path(ks_path.as_posix())

sub = "DO79"
ses = "240109_001"  # "240404_001" # "240109_001"
run_full = False


# Load the data
get_ses_path = lambda toplevel: base_path / toplevel / sub / ses

recording_path = list(get_ses_path("rawdata").glob("**/Record Node 101*"))

assert len(recording_path) == 1, f"{sub} {ses} has unexpected number of recordings."

raw_rec_ = si_extractors.read_openephys(recording_path[0].as_posix(), block_index=0)
probes = gen_probe_group()

raw_rec_ = raw_rec_.set_probegroup(probes)
raw_rec = raw_rec_.split_by("group")[0]  # split the two probes

raw_rec.set_property("group", np.int32(raw_rec.get_probe().shank_ids))
raw_rec = raw_rec.split_by("group")[1]  # split by shank, take one arbitrarily


# preprocess in SI
filtered_rec = si_preprocessing.bandpass_filter(raw_rec, freq_min=300, freq_max=6000)
cmr_rec = si_preprocessing.common_reference(filtered_rec, operator="median")


# preprocess motion correction in SI
motcor_path = get_ses_path("derivatives") / "si_motion_corr"

if run_full:
    motion_correced_rec = si_preprocessing.correct_motion(
        recording=cmr_rec, preset="kilosort_like", folder=motcor_path / "motion_outputs"
    )
    motion_correced_rec.save(folder=motcor_path / "si_recording")


motion_info = si_preprocessing.motion.load_motion_info(
    motcor_path / "motion_outputs"
)
full_plot_motion(get_ses_path("derivatives"), motion_info)


# Run sorting in KS (using KS motion correction)
# run kilosort motion correction no whitening
# TODO: just skip all of KS preprocessing! but OK for now with fork that skips as uses KS directly...

sorter = "kilosort2_5"
if run_full:
    si_sorters.run_sorter(
        sorter_name=sorter,
        recording=cmr_rec,
        car=False,
        freq_min=150,
        output_folder=get_ses_path("derivatives") / sorter,  # TODO: output_folder
    )


# Load kilosort's temp_wh.dat that inclueds the KS-preprocessed data.
sorter_output_path = (
    get_ses_path("derivatives") / sorter / "sorter_output"
)

tmp = sorter_output_path / "temp_wh.dat"

channel_map = np.load(sorter_output_path / "channel_map.npy")

if channel_map.ndim == 2:
    channel_indices = channel_map.ravel()
else:
    assert channel_map.ndim == 1
    channel_indices = channel_map

params = runpy.run_path(sorter_output_path / "params.py")

ks_rec = BinaryRecordingExtractor(
    tmp,
    raw_rec.get_sampling_frequency(),  # TODO: assert against params
    params["dtype"],
    num_channels=channel_indices.size,
    t_starts=None,  # TODO: use from above
    channel_ids=raw_rec.get_channel_ids(),
    time_axis=0,
    file_offset=0,
    gain_to_uV=raw_rec.get_property("gain_to_uV")[channel_indices],
    offset_to_uV=raw_rec.get_property("offset_to_uV")[channel_indices],
    is_filtered=True,
    num_chan=None,
)


# Load back the SI preprocessed data (in case run_full =False).
pp_si_rec = load_extractor(
    get_ses_path("derivatives") / "si_motion_corr" / "si_recording"
)


# Plot all outputs for visual comparison.
plot_list_of_recordings(
    get_ses_path("derivatives"),
    ses,
    raw_rec.get_times(),
    ["filter", "cmr"],  # pp_si_rec, ks_rec
    [filtered_rec, cmr_rec],  # "si_pp", "ks_pp"
)
