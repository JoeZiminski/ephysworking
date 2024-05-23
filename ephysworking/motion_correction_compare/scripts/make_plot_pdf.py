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
from matplotlib.backends.backend_pdf import PdfPages


path_ = Path("/ceph/akrami/Dammy/ephys")
output_path = Path("/ceph/akrami/Dammy/jziminski_temp")

# path_ = Path(r"Y:\Dammy\ephys")
# output_path = Path(r"Y:\Dammy\jziminski_temp")

filepath = output_path / "ephys-traces-pictures.pdf"
if filepath.is_file():
    filepath.unlink()

import matplotlib as mpl
mpl.rcParams.update({'lines.linewidth': 0.25})
mpl.rcParams.update({'font.size': 5})
mpl.rcParams.update({"figure.titlesize": 10})
pdf_pages = PdfPages(filepath)

sessions = [ses_path.name for ses_path in list(path_.glob("DO*"))]
sessions = sorted(sessions)

for ses in sessions:

    if "concat" in ses:
        continue

    preprocessed_path = path_ / ses / "sorting_no_si_drift" / "preprocessed"

    if not preprocessed_path.is_dir():
        continue

    try:
        recording = si.load_extractor(preprocessed_path)
    except ValueError:
        print(f"Could not open {ses}, probably 'not a cached folder'")
        continue

    print(f"Running session: {ses}")

    assert recording.get_num_segments() == 1

    recording_split_by_probe = recording.split_by("group")

    for probe_idx, shank_indexes in enumerate([[0, 1, 2, 3], [4, 5, 6, 7]]):

        recording_probe = recording_split_by_probe[probe_idx]

        recording_probe.set_property(
            "group", np.int32(recording_probe.get_probe().shank_ids)
        )

        recording_probe_split_by_shank = recording_probe.split_by("group")

        for shank_idx in shank_indexes:

            try:
                recording_probe_shank = recording_probe_split_by_shank[shank_idx]
            except KeyError:
                print(f"shank {shank_idx} not found, probe {probe_idx} ID {ses}")
                continue

            fig, ax = plt.subplots(2, 1)
            fig.suptitle(f"{ses}, probe: {probe_idx} shank: {shank_idx}")

            for i, (start_time, stop_time) in enumerate(
                    zip([300, 1500],
                        [301, 1501])
            ):
                try:
                    si_widgets.plot_traces(
                        recording_probe_shank,
                        order_channel_by_depth=True,
                        time_range=(start_time, stop_time),
                        mode="line",
                        show_channel_ids=True,
                        return_scaled=True,
                        ax=ax[i],
                    )
                except ValueError:
                    print(f"Something went from plotting {ses}, probe: {probe_idx} shank: {shank_idx}")
                    plt.close(fig)
                    continue
                ax[i].get_legend().remove()

            pdf_pages.savefig(fig)

            plt.close(fig)

pdf_pages.close()
print("Finished running.")

