"""JAX port of ``WDMDomain`` / ``WDMDomainWrap`` (TDIonTheFly.hh:386).

Passive container for the observed-data WDM coefficients and the
inverse-noise array. The actual inner-product accumulators
(``add_ip_contrib`` / ``add_ip_swap_contrib``) are expressed as JAX
``jnp.sum`` reductions inside the kernels in
:mod:`fastlisaresponse.jax.wdm.computation_group`; this class only
needs to expose the data layout helpers.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class WDMDomainWrapJAX:
    """Mirror of the C++ ``WDMDomainWrap`` data container.

    Constructor signature matches the C++ wrapper exactly:
        (wdm_data, wdm_noise, layer_df, layer_dt,
         Nf, Nt, num_channel, ind_min_t, ind_max_t, ind_min_f, ind_max_f,
         num_data, num_noise)
    """

    def __init__(self,
                 wdm_data, wdm_noise,
                 layer_df: float, layer_dt: float,
                 Nf: int, Nt: int, num_channel: int,
                 ind_min_t: int, ind_max_t: int,
                 ind_min_f: int, ind_max_f: int,
                 num_data: int, num_noise: int):
        self.wdm_data = jnp.asarray(np.asarray(wdm_data))
        self.wdm_noise = jnp.asarray(np.asarray(wdm_noise))
        self.layer_df = float(layer_df)
        self.layer_dt = float(layer_dt)
        self.Nf = int(Nf)
        self.Nt = int(Nt)
        self.num_channel = int(num_channel)
        self.ind_min_t = int(ind_min_t)
        self.ind_max_t = int(ind_max_t)
        self.ind_min_f = int(ind_min_f)
        self.ind_max_f = int(ind_max_f)
        self.Nf_active = self.ind_max_f - self.ind_min_f + 1
        self.Nt_active = self.ind_max_t - self.ind_min_t + 1
        self.num_data = int(num_data)
        self.num_noise = int(num_noise)
