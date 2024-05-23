from pathlib import Path

from ephydworking.inter_session_alignment.utils import pipeline_steps

PLATFORM = "l"  # "w" or "l"
if PLATFORM == "w":
    BASE_PATH = Path(
        r"X:\neuroinformatics\scratch\jziminski\1119617\test_motion_project\derivatives"
    )
elif PLATFORM == "l":
    BASE_PATH = Path(
        "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/test_motion_project/derivatives"
    )

SUB = "sub-013_id-1121381"  # "1119617"
SESSIONS = [
    "ses-006_date-20231223_type-lse2",
    "ses-003_date-20231221_type-pretest",
]  # ["1119617_LSE1_shank12_g0", "1119617_pretest1_shank12_g0", "1119617_posttest1_shank12_g0"]
# all sessions will be aligned to first in list.

if __name__ == "__main__":

    # TODO: maybe 'rigid' option is better, look into interpolation and
    # see if it can be simplified or use spikeinterface API more directly.

    sub_path = BASE_PATH / SUB

    for ses in SESSIONS:

        pipeline_steps.save_motion_histograms(sub_path, ses)

        if ses != SESSIONS[0]:
            pipeline_steps.align_to_first_session(sub_path, SESSIONS[0], ses)

        pipeline_steps.save_data_plots(sub_path, ses)
