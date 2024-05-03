import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.widgets as si_widgets
import probeinterface.plotting as pi_plotting

def save_plot(
    x, y, xlabel, ylabel, output_filepath
):
    """
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(output_filepath)
    plt.close(fig)


def plot_list_of_recordings(
    output_path, ses_name, recording_times, names_list, recording_list, mode="map", clim=None,
):
    """
    """
    output_path.mkdir(exist_ok=True, parents=True)

    quarter_times = np.quantile(
        recording_times, (0, 0.25, 0.5, 0.75, 0.95)
    )  # not quite the end so bin doesn't go over edge

    start_times = [
        np.random.uniform(quarter_times[i], quarter_times[i + 1])
        for i in range(len(quarter_times) - 1)
    ]

    bin_sizes = (0.05, 1, 5)

    for name, rec in zip(names_list, recording_list):

        for start in start_times:
            for bin in bin_sizes:
                fig, ax = plt.subplots()

                si_widgets.plot_traces(
                    rec,
                    order_channel_by_depth=False,
                    time_range=(start, start + bin),
                    ax=ax,
                    mode=mode,
                    show_channel_ids=True,
                    clim=clim,
                )
                format_start = f"{start:0.2f}"
                ax.set_title(
                    f"{ses_name}\n start time: {format_start}, bin size: {bin}"
                )
                fig.savefig(
                    output_path / f"name-{name}_start-{format_start}_bin-{bin}.png"
                )
                plt.close(fig)


def save_probe_plot(probe, output_path, with_contact_id=False, with_device_index=False):
    """
    """
    fig, ax = plt.subplots()
    pi_plotting.plot_probe(probe, ax=ax, with_contact_id=with_contact_id, with_device_index=with_device_index)
    fig.savefig(
        output_path / f"probe_plot.png"
    )



def full_plot_motion(output_path, motion_info):
    """
    """
    save_plot(
        motion_info["temporal_bins"],
        motion_info["motion"],
        "Time (s)",
        "Displacement (μm)",
        output_path / "motion_per_channel.png",
    )

    average_motion_over_channels = np.mean(motion_info["motion"], axis=1)

    save_plot(
        motion_info["temporal_bins"],
        average_motion_over_channels,
        "Time (s)",
        "Displacement (μm)",
        output_path / "avg_motion.png",
    )

    fig, ax = plt.subplots()
    si_widgets.plot_motion(motion_info, figure=fig)
    fig.savefig(output_path / "si_motion_figure.png")
    plt.close(fig)
