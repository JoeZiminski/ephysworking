"""
Save output from inter-session alignment pipeline.
"""

from pathlib import Path

from ephydworking.inter_session_alignment.utils import pipeline_steps

SESSIONS = [
    "1119617_pretest1_shank12_g0"
]  # "1119617_LSE1_shank12_g0", "1119617_pretest1_shank12_g0", "1119617_posttest1_shank12_g0"]

# TODO: how is it handling multi-shank?

if __name__ == "__main__":

    PLATFORM = "w"  # "w" or "l"
    if PLATFORM == "w":
        base_path = Path(
            r"X:\neuroinformatics\scratch\jziminski\1119617\derivatives"
        )
    elif PLATFORM == "l":
        base_path = Path(
            "/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/1119617/derivatives"
        )

    for ses in SESSIONS:

        pipeline_steps.save_data_plots(base_path, ses)
