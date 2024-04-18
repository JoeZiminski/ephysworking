import numpy as np
import probeinterface


def gen_probe_group(probe_name="ASSY-236-P-1"):
    """
    vendored directly from Dammy's code at
    https://github.com/AOONIH/ephys/blob/3a394b2025ee57a079ff2673bc2ce762155c1b5a/sorting_functions.py#L207
    """
    manufacturer = "cambridgeneurotech"

    probe1 = probeinterface.get_probe(manufacturer, probe_name)
    probe1.wiring_to_device("cambridgeneurotech_mini-amp-64")
    # probe1 = probe.copy()
    probe2 = probe1.copy()
    probe2.wiring_to_device("cambridgeneurotech_mini-amp-64")

    probe2.set_device_channel_indices(probe1.device_channel_indices + 64)
    if np.unique(probe1.shank_ids).shape[0] > 1:
        probe2.set_shank_ids((probe1.shank_ids.astype(int) + 4).astype(str))
    # logger.debug(probe1.device_channel_indices,probe2.device_channel_indices)
    probe2.move([5000, 0])
    probes = probeinterface.ProbeGroup()
    probes.add_probe(probe1)
    probes.add_probe(probe2)
    probes.set_global_device_channel_indices(
        np.concatenate([probe1.device_channel_indices, probe2.device_channel_indices])
    )

    return probes
