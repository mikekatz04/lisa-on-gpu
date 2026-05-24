"""JAX-native mirror of ``TDIConfigWrap`` -- a frozen data container.

The C++ ``TDIConfig`` is a passive struct: the projection kernel reads
its arrays (``unit_starts``, ``unit_lengths``, ``tdi_base_link``,
``tdi_link_combinations``, ``tdi_signs_in``, ``channels``,
``num_units``, ``num_channels``) without ever modifying them. We mirror
that 1:1; the arrays stay as numpy ``int32`` / ``float64`` since the
projection iterates over them at trace time and uses their values to
build the JAX-traced computation.
"""
from __future__ import annotations

import numpy as np


class TDIConfigWrapJAX:
    """Pure-Python analog of ``TDIConfigWrap``.

    Accepts the same 8-tuple as the C++ wrapper:
        (unit_starts, unit_lengths, tdi_base_links, tdi_link_combinations,
         tdi_signs, channels, num_units, num_channels)
    """

    def __init__(
        self,
        unit_starts,
        unit_lengths,
        tdi_base_links,
        tdi_link_combinations,
        tdi_signs,
        channels,
        num_units: int,
        num_channels: int,
    ):
        self.unit_starts = np.asarray(unit_starts, dtype=np.int32)
        self.unit_lengths = np.asarray(unit_lengths, dtype=np.int32)
        self.tdi_base_links = np.asarray(tdi_base_links, dtype=np.int32)
        self.tdi_link_combinations = np.asarray(
            tdi_link_combinations, dtype=np.int32
        )
        self.tdi_signs = np.asarray(tdi_signs, dtype=np.float64)
        self.channels = np.asarray(channels, dtype=np.int32)
        self.num_units = int(num_units)
        self.num_channels = int(num_channels)
