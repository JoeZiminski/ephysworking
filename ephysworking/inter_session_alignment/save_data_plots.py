import spikeinterface as si
import matplotlib.pyplot as plt
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion

import numpy as np
from pathlib import Path
from spikeinterface.sortingcomponents import motion_estimation
import histogram_generation
from spikeinterface import widgets

SESSIONS = ["1119617_pretest1_shank12_g0"] # "1119617_LSE1_shank12_g0", "1119617_pretest1_shank12_g0", "1119617_posttest1_shank12_g0"]

# TODO: how is it handling multi-shank?

if __name__ == "__main__":

    PLATFORM = "w"  # "w" or "l"
    if PLATFORM == "w":
        base_path = Path(r"X:\neuroinformatics\scratch\jziminski\1119617\derivatives")
    elif PLATFORM == "l":
        base_path = Path("/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/1119617/derivatives")

    for ses in SESSIONS:

        ses_path = base_path / ses

        for subpath in ["preprocessing", "shifted_data"]:

            recording_path = ses_path / subpath / "si_recording"
            if not recording_path.is_dir():
                continue

            recording = si.load_extractor(recording_path)

            for shank_id, shank_rec in recording.split_by("group").items():

                # assert more than one segment

                # plot shanks separately
                all_times = shank_rec.get_times(segment_index=0)
                quarter_times = np.quantile(all_times, (0, 0.25, 0.5, 0.75, 0.95))  # not quite the end so bin doesn't go over edge
                start_times = [np.random.uniform(quarter_times[i], quarter_times[i + 1]) for i in range(len(quarter_times) - 1)]

                bin_sizes = (0.05, 1, 5)

                for start in start_times:
                    for bin in bin_sizes:

                        fig, ax = plt.subplots()

                        widgets.plot_traces(
                            shank_rec,
                            order_channel_by_depth=True,
                            time_range=(start, start+bin),
                            ax=ax,
                        )
                        format_start = f"{start:0.2f}"
                        ax.set_title(f"{ses}\n start time: {format_start}, bin size: {bin}")
                        fig.savefig(recording_path.parent / f"shank-{shank_id}_start-{format_start}_bin-{bin}.png")
                        plt.close(fig)
