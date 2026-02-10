import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import DefaultOrbits
from lisatools.utils.constants import *

from lisatools.domains import WAVELET_DURATION, TDSignal, TDSettings, FDSignal, FDSettings, WDMSignal, WDMSettings, WDMLookupTable
from fastlisaresponse.gbcomps import GBWDMComputations


if __name__ == "__main__":

    force_backend = "cuda12x"

    xp = np if force_backend == "cpu" else cp
    orbits = DefaultOrbits(force_backend=force_backend)
    orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    dt = 10.0
    Tobs = 2 * YRSID_SI
    Tobs = WAVELET_DURATION * int(Tobs / WAVELET_DURATION)
    Nobs = int(Tobs / dt)
    N_sparse = 256
    t_tdi = xp.linspace(0.0, Tobs, N_sparse + 1)[1:-1]
    Tobs = Nobs * dt

    wdm_settings = WDMSettings(Tobs, dt)
    wdm_lookup_table = WDMLookupTable(wdm_settings, 50, 20, 3)
    
    gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend)

    num_bin = 3

    data_t_arr = xp.arange(Nobs) * dt
    keep = (data_t_arr > t_tdi[0]) & (data_t_arr < t_tdi [-1])
    tdi_t_arr = data_t_arr[keep]

    amp = xp.full(num_bin, 1e-23)
    f0 = xp.full(num_bin, 4.2300812341e-3)
    fdot = xp.full(num_bin, 1e-16)
    fddot = xp.full(num_bin, 0.0)
    phi0 = xp.full(num_bin, 0.892342342342)
    inc = xp.full(num_bin, 1.2309804223)
    psi = xp.full(num_bin, 3.00908098)
    lam = xp.full(num_bin, 4.827342308)
    beta = xp.full(num_bin, -0.50923423)

    gb_gen = GBTDIonTheFly(
        t_tdi, 
        Tobs,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    output = gb_gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    tdi_output = xp.zeros((num_bin, 3, len(data_t_arr))) 

    tdi_output[:, :, keep]= output.eval_tdi(tdi_t_arr)

    # import matplotlib.pyplot as plt
    # plt.plot(data_t_arr[:num_points], tdi_output[0,0])
    # plt.show()

    td_signal = TDSignal(tdi_output[0], settings=TDSettings(tdi_output.shape[-1], dt, force_backend=force_backend))
    if force_backend == "cpu":
        from scipy.signal.windows import tukey
    else:
        from cupyx.scipy.signal.windows import tukey
    # from eryn.prior import uniform_dist
    # num = 11
    # f_arr = uniform_dist(wdm_lookup_table.f_vals[0], wdm_lookup_table.f_vals[-1]).rvs(size=num)
    # fdot_arr = uniform_dist(wdm_lookup_table.fdot_vals[0], wdm_lookup_table.fdot_vals[-1]).rvs(size=num)
    # output = wdm_lookup_table.get_table_coeffs(f_arr, fdot_arr)
    # breakpoint()
    wdm = td_signal.fft(window=tukey(tdi_output.shape[-1], alpha=0.05)).wdmtransform(settings=wdm_settings)
    # fig, ax = wdm.heatmap()

    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZ1SensitivityMatrix
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from copy import deepcopy
    data_res = DataResidualArray(wdm)
    sens_mat = XYZ1SensitivityMatrix(wdm.settings)

    num_container = 5
    acs_list = []
    for i in range(num_container):
        analysis_container = AnalysisContainer(deepcopy(data_res), deepcopy(sens_mat))
        acs_list.append(analysis_container)
    
    acs = AnalysisContainerArray(acs_list, gpus=[0])
    check_ll = acs.likelihood()

    params = xp.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

    test = gb_comps.get_ll_wdm(params, acs, data_index=None, noise_index=None)
    # fig.savefig("check0.png")
    #plt.show()
    # ms = np.arange(wdm.NF)
    # layer_m = int(f0[0] / wdm.df)
    # ms = np.arange(layer_m - 5, layer_m + 5)
    # ms = ms[(ms >= 0) & (ms < wdm.NF)]

    # ns = np.arange(wdm.NT)
    # ms, ns = [tmp.ravel() for tmp in np.meshgrid(ms, ns)]

    # batch_size = 2
    # inds_batch = np.arange(0, len(ms), batch_size)
    # if inds_batch[-1] < len(ms):
    #     inds_batch = np.concatenate([inds_batch, np.array([len(ms)])])

    # for st_ind, end_ind in zip(inds_batch[:-1], inds_batch[1:]):
    #     test = wdm.wavelet(ms[st_ind:end_ind], ns[st_ind:end_ind])
    #     print(end_ind, len(ms))


    breakpoint()