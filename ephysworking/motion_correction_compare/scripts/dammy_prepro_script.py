"""
------
Notes
------

- motion correction issue for these plots is not due to n_jobs
- Preprocessing ordering
  - bad channel detection, CAR and motion correction (check) should be performed per-shank
  - filtering should be performed per-session - filter edge effects are better than filtering over
    uncorrected boundaries before motion correction!
  - Definitely find bad channels before concatenation. Do not want to compute PDF across sessions!
  - When to concat motion: before if motion is small, after if motion is very large.

TODO: need a 'measure' (scalar) for motion change over a session
  questions - contact first vs. contact after
Q: how far apart are these shanks?
  between channels (horizontal: 25um, vertical: 15um), between shanks: 250um.
  so definitely process per-shank
Q: what happens if you motion correct then slide out a bad channel?
TODO: assumes start time is zero (no times vector)
TODO: handle multi-segment case
TODO: should do some asserts that probes match for both recordings
TODO: refactor to classes (session, probe, shank)
TODO: completely reorganise to do in one loop. The benefit of
TODO: assuming num probes always the same here
TODO: think about a better way to show this data.
"""

from dammyephys.sorting_functions import gen_probe_group
import spikeinterface.full as si
import warnings
import numpy as np
import functools
import spikeinterface.preprocessing as si_preprocessing
import spikeinterface.widgets as si_widgets
import matplotlib.pyplot as plt
from pathlib import Path
from spikeinterface import concatenate_recordings
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfMerger


def get_times_to_plot(session_recordings):
    """
    Get times within a session to plot, that are 33% and 66% of the way through
    the recording. Does not handle segments or the new get_times() functionality.
    """
    session_lengths_s = [rec.get_num_samples() / rec.get_sampling_frequency() for rec in session_recordings]
    summed_ses_lengths = np.r_[0, np.cumsum(session_lengths_s)]

    times_to_plot = []
    concat_times_to_plot = []
    for concat_offset, ses_len in zip(summed_ses_lengths, session_lengths_s):

        fixed_ses_times = np.array([round(0.33 * ses_len, -2), round(0.66 * ses_len, -2)])
        times_to_plot.append(fixed_ses_times)

        concat_times_to_plot.append(
            fixed_ses_times + concat_offset
        )

    return session_lengths_s, times_to_plot, concat_times_to_plot

def make_output_path(output_folder, probe_idx, shank_idx):
    return output_folder / f"probe_{probe_idx}" / f"shank_{shank_idx}"


def load_and_set_probe(path_):
    """
    vendored from Dammy
    """
    full_raw_rec = si.read_openephys(path_)
    probes = gen_probe_group()
    full_raw_rec = full_raw_rec.set_probegroup(probes)
    return full_raw_rec


def preprocess_dammy_adjusted(recording, filter_range=(300,9000)):
    """
    vendored from Dammy with minor adjustments
    """
    bad_channels_ids, _ = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channels_ids)
    recording = si.bandpass_filter(recording, freq_min=filter_range[0], freq_max=filter_range[1])
    recording = si.common_reference(recording, reference='global', operator='median')
    return recording, bad_channels_ids


def save_plots_of_recording_to_file(recording, output_filepath, times=None, period_s=1, title=None):
    """
    Given a single recording, save some trace snippets out to a PDF file.
    Possibility to provide a list of times at which to start the snippet alongside
    the length of the period to plot.
    """
    import matplotlib as mpl
    mpl.rcParams.update({'lines.linewidth': 0.25})  # TODO: move this, dont do it globally here.
    mpl.rcParams.update({'font.size': 5})
    mpl.rcParams.update({"figure.titlesize": 10})

    pdf_pages = PdfPages(output_filepath)
    fig, ax = plt.subplots(*(1, 2))

    if times is None:
        max_len = recording.get_num_samples() / recording.get_sampling_frequency()
        times = (max_len / 4) * np.arange(4)[1:]  # TODO: this is stupid...

    for i, (start_time, stop_time) in enumerate(
            zip(times,
                times + period_s)
     ):
        try:
            si_widgets.plot_traces(
                recording,
                order_channel_by_depth=True,
                time_range=(start_time, stop_time),
                mode="line",
                show_channel_ids=True,
                return_scaled=True,
                ax=ax[i],
            )
        except ValueError:
            plt.close(fig)
            pdf_pages.close()
            raise RuntimeError("Plotting traces failed.")

        ax[i].get_legend().remove()

    if title:
        fig.suptitle(title)
    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()


def preprocess_all_sessions_probes_shanks(
        session_recordings, motion_correct=False, whiten=False, save_data=False
):
    """
    Given a list of recordings of different sessions, go through each
    and split by probe / shank then preprocess each shank individually.
    Store in an output dictionary with as dict[session][probe_idx][shank_idx] = recording.

    TODO: add options to motion correct and whiten her (for looking per-session)
    as well as save the data to file.
    """
    num_sessions = len(session_recordings)

    all_preprocessed_shanks = {idx: {} for idx in range(num_sessions)}

    for session_idx, raw_session_rec in enumerate(session_recordings):

        split_by_probe_dict = raw_session_rec.split_by("group")

        probe_ids = split_by_probe_dict.keys()  # init the dict next level
        all_preprocessed_shanks[session_idx] = {idx: {} for idx in probe_ids}

        for probe_idx, raw_rec_single_probe in split_by_probe_dict.items():

            raw_rec_single_probe.set_property(
                "group", np.int32(raw_rec_single_probe.get_probe().shank_ids)
            )

            split_by_shank_dict = raw_rec_single_probe.split_by("group")

            shank_ids = split_by_shank_dict.keys()  # init the dict next level
            all_preprocessed_shanks[session_idx][probe_idx] = {idx: None for idx in shank_ids}

            for shank_idx, raw_rec_single_shank in split_by_shank_dict.items():

                preprocessed_single_shank, bad_channels_ids = preprocess_dammy_adjusted(raw_rec_single_shank)

                if motion_correct:
                    raise NotImplementedError

                if whiten:
                    raise NotImplementedError

                all_preprocessed_shanks[session_idx][probe_idx][shank_idx] = preprocessed_single_shank

                if save_data:
                    raise NotImplementedError

    return all_preprocessed_shanks


def motion_correct_across_sessions(sub_name, session_recordings, all_preprocessed_shanks, output_folder, motion_method, save_data=False):
    """
    Given an all_preprocessed_shanks output from `preprocess_all_sessions_probes_shanks()`
    iterate through all, for each shank concatenating across sessions, slicing to
    common channels, performing motion correction, and saving images of the
    results and trace snippets to PDF.
    """
    # TODO: this does not handle the spikeinterface 't_start' or 'time_vector' construct.

    session_lengths_s, times_to_plot, concat_times_to_plot = get_times_to_plot(session_recordings)

    num_sessions = len(all_preprocessed_shanks)

    for probe_idx in all_preprocessed_shanks[0].keys():

        for shank_idx in all_preprocessed_shanks[0][probe_idx].keys():

            # First, get the preprocessed recordings (bad channe removed, filtered, CAR)
            # Find the common channels across all recordings to be concatenated, and keep
            # only the common channels.
            preprocessed_recs = [
                all_preprocessed_shanks[idx][probe_idx][shank_idx] for idx in
                range(num_sessions)]

            common_channels = functools.reduce(np.intersect1d,
                                               [recording.channel_ids for
                                                recording in
                                                preprocessed_recs])

            sliced_preprocessed_recs = [
                recording.channel_slice(common_channels) for recording in
                preprocessed_recs]

            # Next, save a plot of the un-motion corrected data as PDF
            shank_output_folder = make_output_path(output_folder, probe_idx, shank_idx)

            shank_output_folder.mkdir(exist_ok=True, parents=True)

            for session_idx in range(num_sessions):
                save_plots_of_recording_to_file(
                    sliced_preprocessed_recs[session_idx],
                    shank_output_folder / f"session-{session_idx}_uncorrected_traces.pdf",
                    times=times_to_plot[session_idx],
                    title=f"Sub: {sub_name}, Session: {session_idx}, Probe: {probe_idx}, Shank: {shank_idx}, uncorrected"
                )

            # Now, concatenate these session recordings together and perform motion correct.
            concatenated_recording = concatenate_recordings(
                sliced_preprocessed_recs)

            shank_motion_folder = shank_output_folder / "motion"

            shank_motion_folder.mkdir(exist_ok=True, parents=True)

            concat_rec_corrected = si.correct_motion(
                recording=concatenated_recording,
                preset=motion_method,
                folder=shank_motion_folder,
                )

            if save_data:
                raise NotImplementedError

            # Load the motion info and use spikeinterface plot motion widget.
            # Add some vertical lines to indicate where the session concatenation
            # occured. Save this to PDF.
            motion_info = si.load_motion_info(shank_motion_folder)

            motion_plot = si.plot_motion(motion_info,
                                         color_amplitude=True,
                                         amplitude_cmap='inferno',
                                         scatter_decimate=10)

            for ax_idx in range(4):
                for ses_time in np.cumsum(session_lengths_s)[:-1]:
                    print(session_lengths_s)
                    motion_plot.axes[ax_idx].axvline(x=ses_time, color="k",
                                                     linewidth=1,
                                                     linestyle=":")

            motion_plot.figure.suptitle(
                f"Sub: {sub_name}, Probe: {probe_idx}, Shank: {shank_idx}")

            pdf_pages = PdfPages(shank_motion_folder / "motion_info.pdf")
            pdf_pages.savefig(motion_plot.figure)
            pdf_pages.close()

            # Finally save plots of the motion correct data
            # (at the same timepoint as the uncorrected data)
            # for comparison.
            for session_idx in range(num_sessions):
                save_plots_of_recording_to_file(
                    concat_rec_corrected,
                    shank_motion_folder / f"session-{session_idx}_corrected_traces.pdf",
                    times=concat_times_to_plot[session_idx],
                    title=f"Sub: {sub_name}, Session: {session_idx}, Probe: {probe_idx}, Shank: {shank_idx}, corrected"
                )


def preprocess_single_subject(
  sub_name, session_paths, output_path, motion_method, n_jobs
):
    """
    For the given sessions, perform the whole preprocessing pipeline:
    - detect bad channel, bandpass filter, common reference per-shank
    - per shank, slice recordings to common good channels and motion correct
    - save plots of trace snippets and motion correction output to PDF.
    At the end of this function collate all the saved PDF into one
    big PDF for readability.
    """
    output_folder = Path(output_path) / sub_name / motion_method

    si.set_global_job_kwargs(n_jobs=n_jobs)

    session_recordings = [load_and_set_probe(path_) for path_ in session_paths]

    # Perform the first few preprocessing steps per-session
    # (detect and remove bad channels, filter, CAR)
    all_preprocessed_shanks = preprocess_all_sessions_probes_shanks(
            session_recordings, motion_correct=False, whiten=False, save_data=False
    )

    # Perform motion correction, saving plots of raw data before and
    # after correction, and the motion plots (and `motion_info`).
    motion_correct_across_sessions(
        sub_name, session_recordings, all_preprocessed_shanks, output_folder, motion_method
    )

    # Merge all the generated PDFs into a single report for the subject.
    pdf_paths_to_merge = []
    for probe_idx in all_preprocessed_shanks[0].keys():
        for shank_idx in all_preprocessed_shanks[0][probe_idx].keys():

            shank_output_folder = make_output_path(output_folder, probe_idx, shank_idx)

            shank_motion_folder = shank_output_folder / "motion"

            pdf_paths_to_merge.append(
                shank_motion_folder / "motion_info.pdf"
            )

            num_sessions = len(session_recordings)
            for session_idx in range(num_sessions):

                pdf_paths_to_merge.append(
                    shank_output_folder /  f"session-{session_idx}_uncorrected_traces.pdf"  # TODO: store name
                )
                pdf_paths_to_merge.append(
                    shank_motion_folder / f"session-{session_idx}_corrected_traces.pdf",
                )

    merger = PdfMerger()

    for pdf in pdf_paths_to_merge:
        merger.append(pdf)

    merger.write(output_folder / f"motion_report_{motion_method}.pdf")
    merger.close()
