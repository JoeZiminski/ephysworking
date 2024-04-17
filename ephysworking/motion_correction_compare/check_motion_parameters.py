# This pipeline does not phase shift to keep things as
# close as possible between SI and Kilosort


# TOOD: does spikeinterface not include scipy? sklearn

# run preprocessing steps in kilosort directly
# INSTRUCTIONS
from spikeinterface import extractors as si_extractors
from spikeinterface import preprocessing as si_preprocessing
from spikeinterface import sorters as si_sorters
import probeinterface as pi
import os
from spikeinterface.core import BinaryRecordingExtractor
import runpy
from pathlib import Path
import platform
import numpy as np
from spikeinterface import load_extractor
import matplotlib.pyplot as plt
import spikeinterface.widgets as si_widgets  # TODO: make consistent
import spikeinterface as si
from ephysworking.utils import save_plot
from ephysworking.motion_correction_compare.utils import gen_probe_group

if platform.system() == "Windows":
    base_path = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\dammy")
    ks_path = Path("X:\neuroinformatics\scratch\jziminski\git-repos\forks\Kilosort_2-5_nowhiten")
else:
    base_path = Path(r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/dammy")
    ks_path = Path("/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/git-repos/forks/Kilosort_2-5_nowhiten")

# os.environ["KILOSORT2_5_PATH"] = ks_path.as_posix()
from spikeinterface.sorters import Kilosort2_5Sorter
Kilosort2_5Sorter.set_kilosort2_5_path(ks_path.as_posix())

sub = "DO79"
ses = "240109_001" # "240404_001" # "240109_001"

options = [
    "si_motion_corr",
    "ks_motion_corr_in_si",
]

get_ses_path = lambda toplevel: base_path / toplevel / sub / ses

recording_path = list(get_ses_path("rawdata").glob("**/Record Node 101*"))

assert len(recording_path) == 1, f"{sub} {ses} has unexpected number of recordings."

raw_rec = si_extractors.read_openephys(recording_path[0].as_posix(), block_index=0)
probes = gen_probe_group()

raw_rec = raw_rec.set_probegroup(probes)

raw_rec = raw_rec.split_by("group")[0]

filtered_rec = si_preprocessing.bandpass_filter(raw_rec, freq_min=300, freq_max=6000)
cmr_rec = si_preprocessing.common_reference(filtered_rec, operator="median")

# on train -
# message Brandon
# Whatsapps


motcor_path = get_ses_path("derivatives") / "si_motion_corr"

if False:
    motion_correced_rec = si_preprocessing.correct_motion(
        recording=cmr_rec, preset="kilosort_like", folder=motcor_path / "motion_outputs"
    )
    motion_correced_rec.save(folder=motcor_path / "si_recording")


# TODO: open issue for load_motion_info docs wrong

# si_widgets.plot_motion(motion_info, ax=ax)  # TODO  is broken see SI issue #
motion_info = si_preprocessing.motion.load_motion_info(motcor_path / "motion_outputs")

save_plot(
    motion_info["temporal_bins"],
    motion_info["motion"],
    "Time (s)",
    "Displacement (μm)",
    motcor_path / "motion_per_channel.png"
)

average_motion_over_channels = np.mean(motion_info["motion"], axis=1)

save_plot(
    motion_info["temporal_bins"],
    average_motion_over_channels,
    "Time (s)",
    "Displacement (μm)",
     motcor_path / "avg_motion.png"
)

sorter = "kilosort2_5"
# if False:
# run kilosort motion correction no whitening
si_sorters.run_sorter(
    sorter_name=sorter, recording=cmr_rec, car=False, freq_min=150, output_folder=get_ses_path("derivatives") / sorter   # TODO: output_folder
)
# TODO: just skip all of KS preprocessing! but OK for now with fork that skips as uses KS directly...

breakpoint()

fig, ax = plt.subplots()

si_widgets.plot_traces(rec, order_channel_by_depth=True,
                       time_range=(start, start + bin), ax=ax, mode="map")
format_start = f"{start:0.2f}"
ax.set_title(f"{ses}\n start time: {format_start}, bin size: {bin}")
fig.savefig(
    get_ses_path("derivatives") / f"name-{name}_start-{format_start}_bin-{bin}.png")
plt.close(fig)

sorter_output_path = get_ses_path("derivatives") / sorter/ "sorter_output"  # SAVE PATH Path(r"X:\neuroinformatics\scratch\jziminski\ephys\dammy\kilosort2_5_output\sorter_output")

tmp = sorter_output_path / "temp_wh.dat"

channel_map = np.load(sorter_output_path / "channel_map.npy")

if channel_map.ndim == 2:
    channel_indices = channel_map.ravel()
else:
    assert channel_map.ndim == 1
    channel_indices = channel_map

params = runpy.run_path(sorter_output_path / "params.py")
# assert len against raw rec

pp_si_rec = load_extractor(
    get_ses_path("derivatives") / "si_motion_corr" / "si_recording"
)

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

# TODO: DIRECT COPY AND PASTE
# TODO: OWN FUNCTION!
all_times = raw_rec.get_times()
quarter_times = np.quantile(all_times, (0, 0.25, 0.5, 0.75, 0.95))  # not quite the end so bin doesn't go over edge
start_times = [np.random.uniform(quarter_times[i], quarter_times[i + 1]) for i in range(len(quarter_times) - 1)]

bin_sizes = (0.05, 1, 5)

for name, rec in zip(["filter", "cmr", "pp_si", "pp_sk"], [filtered_rec, cmr_rec, pp_si_rec, ks_rec]):

    for start in start_times:
        for bin in bin_sizes:
            fig, ax = plt.subplots()

            si_widgets.plot_traces(rec, order_channel_by_depth=True, time_range=(start, start + bin), ax=ax, mode="map")
            format_start = f"{start:0.2f}"
            ax.set_title(f"{ses}\n start time: {format_start}, bin size: {bin}")
            fig.savefig(get_ses_path("derivatives") / f"name-{name}_start-{format_start}_bin-{bin}.png")
            plt.close(fig)
