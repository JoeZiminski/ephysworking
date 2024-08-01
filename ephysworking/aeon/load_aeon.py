from probeinterface.io import _read_imro_string
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np
from probeinterface.plotting import plot_probe
from pathlib import Path

# Set up matplotlib for saving TIFF
import matplotlib as mpl
mpl.rcParams.update({'font.size': 2})
mpl.rcParams.update({"figure.titlesize": 16})

# Options
show_probe = False
data_base_path = Path(rf"X:\neuroinformatics\scratch\jziminski\ephys\inter-session-alignment\histogram_estimation\test_data\aeon-6d-session")

# Load the probe
# TODO: replace with the channel map, including 'set_channel_locations() below.

imDatPrb_pn = "NP2010"
imro_table = "(24,384)"
for i in range(384):
    imro_table += f"({i} 0 0 0 {i})"
probe = _read_imro_string(imro_str=imro_table, imDatPrb_pn=imDatPrb_pn)
probe.set_device_channel_indices(np.arange(probe.get_contact_count()))

if show_probe:
    plot_probe(probe)
    plt.show()


# Load, preprocess and plot recordings
num_chan = 384
fs = 30000

fig, ax = plt.subplots(4, 3)

all_recordings = []
for row, all_rec_nums in enumerate(((10, 11, 12), (50, 51, 52), (100, 101, 102), (133, 134, 135))):
    for col, rec_num in enumerate(all_rec_nums):

        filename = f"NeuropixelsV2Beta_ProbeA_AmplifierData_{rec_num}.bin"
        recording = si.read_binary(data_base_path / filename,
                       sampling_frequency=fs,
                       dtype=np.int16,
                       num_channels=384)

        print(filename)

        recording.set_probe(probe)
        recording.set_channel_locations(probe.contact_positions)

        # recording = si.phase_shift(recording) TODO: need inter_sample_shift from channel map
        recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

        groups = [recording.get_channel_ids()[::2], recording.get_channel_ids()[1::2]]  # makes hardly any difference
        recording = si.common_reference(recording, operator="median", groups=groups)

        si.plot_traces(recording, ax=ax[row, col])
        ax[row, col].set_title(f"{rec_num}")

        all_recordings.append(recording)

fig.savefig(data_base_path / 'data-overview.tiff', dpi=600)


