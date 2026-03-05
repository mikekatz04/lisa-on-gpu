from few.utils.utility import get_polarization_angle, get_viewing_angles
from typing import Optional

from gpubackendtools.interpolate import CubicSplineInterpolant
from fastlisaresponse.tdionfly import TDTDIonTheFly
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py
import numpy as np
try:
    import cupy as xp
    backend = 'cuda12x'
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    backend = 'cpu'

# from lisatools.globalfit.preprocessing import L1ProcessingStep
from preprocessing import L1ProcessingStep
from phentax.waveform import IMRPhenomTHM
from lisatools.detector import L1Orbits

from lisaconstants import ASTRONOMICAL_YEAR
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "notebook"])

FIGSIZE = (10, 6)


import logging
import sys

# Configure logging to display messages in the notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


# ...existing code...
def ecliptic_to_icrs(lambda_ecl, beta_ecl):
    """
    Convert ecliptic coordinates (lambda, beta) in radians to ICRS (RA, Dec) in radians.

    Parameters
    ----------
    lambda_ecl, beta_ecl : float or array-like
        Ecliptic longitude (lambda) and latitude (beta) in radians.

    Returns
    -------
    ra, dec : tuple
        Right ascension and declination in radians (same shape as inputs).
    """
    ecl = SkyCoord(lon=lambda_ecl * u.rad, lat=beta_ecl * u.rad, frame='barycentrictrueecliptic')
    icrs = ecl.transform_to('icrs')
    return icrs.ra.rad, icrs.dec.rad

class EMRITDIonFly:

    def __init__(
        self,
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0
    ):
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.T = Tobs
        self.t0 = t0
    
    @property
    def dt(self) -> float:
        """dt value from data."""
        return self._dt
    
    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self.sampling_frequency = 1 / dt

    def __call__(self, 
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        x0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        Phi_phi0: float,
        Phi_theta0: float,
        Phi_r0: float,
        *add_args: Optional[tuple],
        include_minus_mkn=True,
        **kwargs: Optional[dict],
    ):
        theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
        psi = get_polarization_angle(qS, phiS, qK, phiK)

        lam = phiS
        beta = np.pi / 2 - qS
        
        ra, dec = ecliptic_to_icrs(lam, beta)
        # breakpoint()
        # psi = np.pi - psi
        Kerr_wave = self.wave_gen(
            m1,
            m2,
            a,
            p0,
            e0,
            x0,
            theta,
            phi,
            dist=dist,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=self.T,
            dt=self.dt,
            return_sparse_holder=True,
            include_minus_mkn=include_minus_mkn
        )

        mode_amp_phase = np.unwrap(np.angle(Kerr_wave.teuk_modes), axis=0)
        mode_amp_amp = np.abs(Kerr_wave.teuk_modes)

        ylm_phase = np.angle(Kerr_wave.ylms)
        ylm_amp = np.abs(Kerr_wave.ylms)
        _mode_phase = (
            Kerr_wave.ms[None, :] * Kerr_wave.phases[:, 0][:, None]
            + Kerr_wave.ks[None, :] *  Kerr_wave.phases[:, 1][:, None]
            + Kerr_wave.ns[None, :] *  Kerr_wave.phases[:, 2][:, None]
        )

        AMP_FACTOR = 1/2. # CHECK THIS
        if include_minus_mkn:
            # m >= 0
            keep_minus_m = Kerr_wave.ms != 0
            phase_m_zero_and_above = (_mode_phase - ylm_phase[:Kerr_wave.ms.shape[0]] - mode_amp_phase)
            phase_m_below_zero = -phase_m_zero_and_above[:, keep_minus_m]
            mode_phase = np.concatenate([phase_m_zero_and_above, phase_m_below_zero], axis=-1).T

            amp_m_zero_and_above = AMP_FACTOR * mode_amp_amp * ylm_amp[:_mode_phase.shape[1]]    
            amp_m_below_zero = (AMP_FACTOR * mode_amp_amp * ylm_amp[_mode_phase.shape[1]:])[:, keep_minus_m]
            mode_amp = np.concatenate([amp_m_zero_and_above, amp_m_below_zero], axis=-1).T
        
        else:
            mode_phase = (_mode_phase - ylm_phase[:Kerr_wave.ms.shape[0]] - mode_amp_phase).T
            mode_amp = AMP_FACTOR * (mode_amp_amp * ylm_amp[:Kerr_wave.ms.shape[0]]).T

        t_arr_in = self.t0 + np.repeat(Kerr_wave.t_arr[:, None], mode_phase.shape[0], axis=-1).T
        t_arr_tdi = t_arr_in[:, 1:-1]
        dt = 1.0
        sampling_frequency = 1 / dt
        num_sub = mode_amp.shape[0]

        self.tdi_gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, sampling_frequency, num_sub, t_input=t_arr_in, tdi_config=self.tdi_config, orbits=self.orbits)

        inc = np.zeros(num_sub)
        psi_in = np.full(num_sub, psi)
        ra_in = np.full(num_sub, ra)
        dec_in = np.full(num_sub, dec)
        # beta = np.full(num_sub, qS)
        # output_tdi_fly = self.tdi_gen(inc, psi_in, lam, beta, return_spline=True)
        output_tdi_fly = self.tdi_gen(inc, psi_in, ra_in, dec_in, return_spline=True)

        return output_tdi_fly
        


path = "/Users/mlkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"

source_types = [
    #'noise', 
    #'mbhb', 
    #  'gb', 
    #  'vgb', 
      'emri', 
    #  'sobhb'
    ]
ID = 3

loader = L1ProcessingStep(
    L1_folder=path,
    source_types=source_types,
    source_ids=dict(emri=list(range(8))[ID:ID+1]),
    orbits_class=L1Orbits,
    orbits_kwargs=dict(force_backend=backend, frame="ecliptic"),
    verbose=True,
    do_plots=False
)

full_catalogue = loader.catalogue['EMRI']


parameter_list = full_catalogue[ID].keys()
print("Available parameters in the catalogue:")
for param in parameter_list:
    print(' ->  ' + param)

target_parameters = ['SecondaryMassSSBFrame', 'SecondaryMassSourceFrame']
for id in range(8)[ID:ID+1]:
    for target_parameter in target_parameters:
        print(f"Source ID: {id}, {target_parameter}: {full_catalogue[id][target_parameter]}")
    print("\n")
 

# plt.figure(figsize=FIGSIZE)
# plt.plot(loader.times, loader.data[0], label='L1 X')

# plt.xlabel("Time [s]")
# plt.ylabel("Strain")
# plt.title("LISA TDI X channel - EMRI source #{}".format(ID))
# plt.legend(loc='upper left')
# plt.show()

from fastlisaresponse import ResponseWrapper, pyResponseTDI
import few
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from scipy.signal.windows import tukey

from lisagwresponse import (
    GalacticBinary,
    ReadResponse,
    ReadStrain,
    Response,
    ResponseFromStrain,
    VerificationBinary,
)

cfg = few.get_config()

binary_params = loader.catalogue['EMRI'][ID]
orbits = loader.orbits 
# orbits_file = "orbits.h5" 
orbits_file = '/Users/mlkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/data/EMRI/L1/EMRI_731d_2.5s_L1_source1_0_20251203T225452097594Z.h5'
from h5py import Dataset, File
# orbits_file = "../lisa-on-gpu/new_orbits_file_eq_arm.h5"
# with File(orbits_file) as orbitf:
#     t0 = orbitf.attrs["t0"]

# t0 = orbits.ltt_t0

# from lisatools.detector import Orbits
# orbits = Orbits(orbits_file, t0=t0)
# orbits.configure(linear_interp_setup=True)

# check = orbits.get_light_travel_times(np.array([orbits.ltt_t0, 10001.0044334]), 12)

Tplunge = binary_params['TimeCoalescenceSSBFrame'] / ASTRONOMICAL_YEAR
Tobs = np.ceil(Tplunge)

T_stop = Tplunge  # 29969888.148967054

print(f"Observation time: {Tplunge} years")
dt = 2.5

##======================= Waveform set-up arguments from validation notebook https://gitlab.esa.int/lisa-sgs/sim/lisasimdatavalid/-/blob/main/notebooks/valid_level3/EMRI/ccomparison_fastlisaresponse.ipynb?ref_type=heads

sum_kwargs = {
    "pad_output": True,
}

from lisatools.utils.constants import YRSID_SI

N_pts = 16384

inspiral_kwargs_main = {
    "DENSE_STEPPING": 0,  # sparsely sampled trajectory
    "max_init_len": int(1e4),  # length of trajectories well under 1000
}

inspiral_kwargs_tof = {
    "DENSE_STEPPING": 0,  # sparsely sampled trajectory
    "max_init_len": int(1e4),  # length of trajectories well under 1000
    "upsample": True, 
    "new_t": np.linspace(0.0 + 100.0, binary_params['TimeCoalescenceSSBFrame'] - 100.0, N_pts),
}

amplitude_kwargs = {
    # "max_init_len": int(1e8),  # all of the trajectories will be well under len = 1000
    # "use_gpu": True,
    # "file_dir":"/data/leuven/367/vsc36785/LISA/FastEMRIWaveforms/data"
}

mode_selector_kwargs = {
    'mode_selection_threshold': 1e-6
    # "mode_selection": "all",  # modes_tmp,
}

##======================= Response set-up 

index_lambda = 8
index_beta = 7
waveform_model = 'Kerr'

from fastlisaresponse.tdiconfig import TDIConfig
t_buffer = 30000.0
response_kwargs = {
    'Tobs': Tobs,
    'dt': dt,
    'index_lambda': index_lambda,
    'index_beta': index_beta,
    'flip_hx': True,
    'force_backend': backend,
    'tdi': TDIConfig('2nd generation'),
    'tdi_chan': 'XYZ',
    'order': 40,
    'remove_garbage': False,
    'is_ecliptic_latitude': False,
    't_buffer': t_buffer,
}

few_generator = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    return_list=False,    # returns hp - i*hx as a complex cupy array
    inspiral_kwargs=inspiral_kwargs_main,
    sum_kwargs=sum_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    frame="detector",
    mode_selector_kwargs=mode_selector_kwargs,
    force_backend=backend
)

orbits_kwargs = orbits.kwargs
orbits_kwargs['frame'] = 'icrs'

icrs_orbits = L1Orbits(**orbits_kwargs)

print(binary_params.keys())
print('')

tdi_generator = ResponseWrapper(
    few_generator,
    orbits=icrs_orbits,
    **response_kwargs
)



def icrs_to_ecliptic(ra, dec):
    """Convert ICRS coordinates (ra, dec) to ecliptic coordinates (lambda, beta)."""


    icrs_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
    ecliptic_coord = icrs_coord.barycentrictrueecliptic

    lambda_ecl = ecliptic_coord.lon.rad
    beta_ecl = ecliptic_coord.lat.rad

    return lambda_ecl, beta_ecl


m1 = binary_params['PrimaryMassSSBFrame']
m2 = binary_params['SecondaryMassSSBFrame']
a = binary_params['PrimarySpinParameter']
p0 = binary_params['SemiLatusRectum']
e0 = binary_params['Eccentricity']
xI0= np.cos(binary_params['InclinationAngle'])
dist = binary_params['LuminosityDistance'] * 1e-3 # convert to Gpc
ra = binary_params['RightAscension']
dec = binary_params['Declination']
lambda_ecl, beta_ecl = icrs_to_ecliptic(ra, dec)

if True:  # tdi_generator.response_model.tdi_orbits.frame == 'ecliptic':
    print('using ecliptic coordinates for source position')
    qS = np.pi / 2 - beta_ecl
    phiS = lambda_ecl
else:
    print('using icrs coordinates for source position')
    # these are probably not correct - need to check, but it doesnt work anyway
    qS = dec
    phiS = ra

qK = binary_params['PolarAnglePrimarySpin']
phiK = binary_params['AzimuthalAnglePrimarySpin']


Phi_phi0 = binary_params['AzimuthalPhase']
Phi_theta0 = binary_params['PolarPhase']
Phi_r0 = binary_params['RadialPhase']


# now set up the time array
tc = binary_params["TimeCoalescenceSSBFrame"]
ind_0 = - tc // dt 
ind_f = ind_0 + int(response_kwargs['Tobs'] * ASTRONOMICAL_YEAR / dt)
waveform_time = np.arange(ind_0, ind_f) * dt

times_waveform = waveform_time - ind_0 * dt + binary_params["TimeReferenceSSBFrame"]

wf_params = np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])
print("WF_PARAMS:", wf_params)
t0 = times_waveform[0]
del tdi_generator
t0_start = 1000.0
tdi_generator = ResponseWrapper(
    few_generator,
    orbits=icrs_orbits,
    t0=t0,
    **response_kwargs
)

include_minus = True
# h = few_generator(*wf_params, include_minus_mkn=include_minus, T=tdi_generator.Tobs, dt=tdi_generator.dt)
# _tmp = 100000


dt = 2.5
# tmp1 = ReadStrain(t0 + np.arange(len(h)) * dt, h.real, -h.imag,  orbits=orbits_file, strain_interp_order=5, ra=ra, dec=dec)
# tmp1.write("file_tmp_11.h5", dt=dt, size=h[:_tmp].shape[0], t0=t0 + t0_start, mode="w",)
tdi_channels = tdi_generator(*wf_params, include_minus_mkn=include_minus)

# import h5py
# with h5py.File("file_tmp_11.h5", "r") as fp:
#     tmp_dat = fp["tcb"]["y"][:]

# t1 = np.arange(t0 + t0_start, t0 + t0_start + len(tmp_dat) * dt, dt)
# fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
# ax = axs.ravel()
# for i in range(6):
#     ax[i].plot(t0 + np.arange(tdi_generator.response_model.y_gw[i].shape[0]) * tdi_generator.dt, tdi_generator.response_model.y_gw[i])
#     ax[i].plot(t1, tmp_dat[:, i], "--")
#     # ax[i].set_xlim(50000.0, 60000.0)
#     # ax[i].set_ylim(-1.e-22, 1.e-22)
# plt.show()

# breakpoint()
modes_here = [tuple(tmp) for tmp in np.array([
    few_generator.waveform_generator.ls,
    few_generator.waveform_generator.ms,
    few_generator.waveform_generator.ks,
    few_generator.waveform_generator.ns,
]).T]



mode_selector_kwargs_tof = {
    'mode_selection': modes_here
    # "mode_selection": 1e-5,  # "all"
}
wave_generator_Kerr = FastKerrEccentricEquatorialFlux(
    force_backend=backend,
    inspiral_kwargs=inspiral_kwargs_tof,
    sum_kwargs=sum_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    mode_selector_kwargs=mode_selector_kwargs_tof,
)

# create a 3 panel plot, 1 above and 2 below wide half of the above
do_window = True
remove_garbage = False

X_channel_waveform = tdi_channels[0]  # .get()

if remove_garbage:
    gar_time = response_kwargs['t_buffer']
    gar_samples = int(gar_time / dt)
    X_channel_waveform = X_channel_waveform[gar_samples:-gar_samples]
    times_waveform_here = times_waveform[gar_samples:-gar_samples]
else:
    times_waveform_here = times_waveform

if do_window:
    window = tukey(len(X_channel_waveform), alpha=0.01)
else:
    window = np.ones(len(X_channel_waveform))

X_channel_waveform = X_channel_waveform * window

from matplotlib.gridspec import GridSpec

xlim_full = (t0, tc + t0)

ylim_full = (-1.5e-22, 1.5e-22)
ylim_start = (-1.2e-23, 1.2e-23)
ylim_end = (-7e-23, 6e-23)

fly_kwargs = {
    'Tobs': response_kwargs['Tobs'],
    'dt': response_kwargs['dt'],
    # 'force_backend': response_kwargs['force_backend'],
    'tdi': response_kwargs["tdi"],
    'tdi_chan': 'XYZ',
}
emri_gen = EMRITDIonFly(
    wave_generator_Kerr,
    icrs_orbits,
    TDIConfig('2nd generation'),
    dt,
    Tobs,
    t0,
)

test = emri_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, include_minus_mkn=include_minus)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig)

# Top panel spanning both columns
ax1 = fig.add_subplot(gs[0, :])

# Bottom two panels side by side
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(loader.times, loader.data[0], label='L1 X')
ax1.plot(times_waveform_here, X_channel_waveform, alpha=0.4, label='EMRI waveform X', c='k', ls='--')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("X channel")
ax1.set_title("LISA TDI X channel - EMRI source with FastLISAResponse")
ax1.legend(loc ='upper left')
# ax1.set_xlim(xlim_full)
# ax1.set_ylim(ylim_full)

#----------------------------------------------------------------
new_t = times_waveform.copy()  # [slice_inds]

xlim_start = (np.maximum(test.t_arr[0, 0], new_t[0]), np.minimum(test.t_arr[0, -1], new_t[-1]))
new_t = new_t[(new_t > xlim_start[0]) & (new_t < xlim_start[1])]
new_t = new_t[(new_t > test.t_arr.min().item()) & (new_t < test.t_arr.max().item())]
new_t = new_t[500000:510000]
xlims_set_2 = (new_t[0], new_t[-1])
new_vals = test.eval_tdi(new_t)
# # new_freq_vals = test.phase_ref_spl(np.tile(new_t, (test.num_bin, 1)), derivative=1) / 2 * np.pi
# # new_power_vals = test.tdi_amp_spl(np.tile(new_t, (test.num_bin, 3, 1))) ** 2
new_wave = np.sum(new_vals, axis=0)[0]

ax2.plot(loader.times, loader.data[0], label='L1 X')
ax2.plot(times_waveform_here, X_channel_waveform, alpha=0.7, label='EMRI waveform X', c='k')  # , ls='--')
ax2.plot(new_t, new_wave, alpha=0.7, label='EMRI waveform X', c='r', ls='--')
ax2.set_xlim(xlims_set_2)
# ax2.set_ylim(ylim_start)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("X channel")


#----------------------------------------------------------------
new_t = times_waveform.copy()  # [slice_inds]
xlim_end = (np.maximum(test.t_arr[0, 0], new_t[0]), np.minimum(test.t_arr[0, -1], new_t[-1]))
new_t = new_t[(new_t > xlim_end[0]) & (new_t < xlim_end[1])]
new_t = new_t[(new_t > test.t_arr.min().item()) & (new_t < test.t_arr.max().item())]
new_t = new_t[-510000:-500000]
xlims_set_3 = (new_t[0], new_t[-1])

new_vals = test.eval_tdi(new_t)
# # new_freq_vals = test.phase_ref_spl(np.tile(new_t, (test.num_bin, 1)), derivative=1) / 2 * np.pi
# # new_power_vals = test.tdi_amp_spl(np.tile(new_t, (test.num_bin, 3, 1))) ** 2
new_wave = np.sum(new_vals, axis=0)[0]


ax3.plot(loader.times, loader.data[0], label='L1 X')
ax3.plot(times_waveform_here, X_channel_waveform, alpha=0.7, label='EMRI waveform X', c='k')  # , ls='--')
ax3.plot(new_t, new_wave, alpha=0.7, label='EMRI waveform X', c='r', ls='--')
ax3.set_xlim(xlims_set_3)
# ax3.set_ylim(ylim_end)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("X channel")

plt.tight_layout()
# fig.savefig(f"check_emri_{ID}.png")
plt.show()
breakpoint()
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import SensitivityMatrix

from lisatools.domains import TDSettings, FDSettings, FDSignal, TDSignal


# select the part of the signal above t0

mask = slice(None) #slice(gar_samples, -gar_samples) if remove_garbage else slice(None)

times_waveform_here = times_waveform[mask]

tdi_channels_here = np.array([tdi_channel[mask] for tdi_channel in tdi_channels])

# now spline interpolate the tdi channels to the same time array as the data
from scipy.interpolate import CubicSpline
window = tukey(len(times_waveform_here), alpha=0.01)

xyz_splined = np.array([
    CubicSpline(times_waveform_here, tdi_channel_here)(loader.times) for tdi_channel_here in tdi_channels_here
])

NPLOT = int(1e9)

plt.plot(loader.times[:NPLOT], loader.data[0][:NPLOT], label='L1 X')
plt.plot(loader.times[:NPLOT], xyz_splined[0][:NPLOT], alpha=0.7, label='EMRI waveform X', c='k', ls='--')


