import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "TDIonTheFly.hh":
    ctypedef void* cmplx 'cmplx'
    cdef cppclass GBTDIonTheFlyWrap "GBTDIonTheFly":
        GBTDIonTheFlyWrap(double T_) except+
        void dealloc() except+
        void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        void add_tdi_config(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_) except+
        void run_wave_tdi(
            void *buffer, int buffer_length,
            cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase,
            double *phi_ref, double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels) except+
        int get_gb_buffer_size(int N) except+

    cdef cppclass TDSplineTDIWaveformWrap "TDSplineTDIWaveform":
        # TDSplineTDIWaveformWrap(CubicSplineWrap *amp_spline_, CubicSplineWrap *phase_spline_) except+
        void run_wave_tdi(
            void *buffer, int buffer_length,
            cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase,
            double *phi_ref, double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels) except+
        int get_td_spline_buffer_size(int N) except+
        void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        void add_tdi_config(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_) except+
        void add_amp_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_) except+
        void add_phase_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_) except+
        void dealloc() except+
        void check_x() except+ 
        void print_orbits_tdi() except+

    cdef cppclass FDSplineTDIWaveformWrap "FDSplineTDIWaveform":
        # FDSplineTDIWaveformWrap(CubicSplineWrap *amp_spline_, CubicSplineWrap *freq_spline_, double *phase_ref_) except+
        void run_wave_tdi(
            void *buffer, int buffer_length,
            cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase,
            double *phi_ref, double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels) except+
        int get_fd_spline_buffer_size(int N) except+
        void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        void add_tdi_config(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_) except+
        void add_amp_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_) except+
        void add_freq_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_) except+
        void dealloc() except+


cdef class pyGBTDIonTheFly:
    cdef GBTDIonTheFlyWrap *g
    cdef double T

    def __cinit__(self, T):
        self.T = T 
        
        self.g = new GBTDIonTheFlyWrap(self.T)

    def add_orbit_information(self, *args, **kwargs):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
            links,
            sc_r, 
            sc_e,
            armlength
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g.add_orbit_information(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in,
            armlength
        )

    def add_tdi_config(self, *args, **kwargs):
        (
            unit_starts,
            unit_lengths,
            tdi_base_link,
            tdi_link_combinations,
            tdi_signs_in,
            channels,
            num_units,
            num_channels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t unit_starts_in = unit_starts
        cdef size_t unit_lengths_in = unit_lengths
        cdef size_t tdi_base_link_in = tdi_base_link
        cdef size_t tdi_link_combinations_in = tdi_link_combinations
        cdef size_t tdi_signs_in_in = tdi_signs_in
        cdef size_t channels_in = channels
        
        self.g.add_tdi_config(
            <int *>unit_starts_in, 
            <int *>unit_lengths_in, 
            <int *>tdi_base_link_in, 
            <int *>tdi_link_combinations_in, 
            <double *>tdi_signs_in_in, 
            <int *>channels_in, 
            num_units, 
            num_channels
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild, (self.T,))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_length,
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phi_ref,
            params, t_arr,
            N, num_sub, n_params, nchannels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t buffer_in = buffer
        cdef size_t tdi_channels_arr_in = tdi_channels_arr
        cdef size_t tdi_amp_in = tdi_amp
        cdef size_t tdi_phase_in = tdi_phase
        cdef size_t phi_ref_in = phi_ref
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr

        self.g.run_wave_tdi(
            <void *>buffer_in, buffer_length,
            <cmplx *> tdi_channels_arr_in, 
        <double *> tdi_amp_in, <double *> tdi_phase_in,
        <double*> phi_ref_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params, nchannels)

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_gb_buffer_size(N)

def rebuild(
    T,
):
    c = pyGBTDIonTheFly(
        T,
    )
    return c


cdef class pyTDSplineTDIWaveform:
    cdef TDSplineTDIWaveformWrap *g
    
    def __cinit__(self):
        self.g = new TDSplineTDIWaveformWrap()

    def print_orbits_tdi(self):
        self.g.print_orbits_tdi()
        
    def add_amp_spline(self, *args, **kwargs):

        (x0, y0, c1, c2, c3,  ninterps, length, spline_type), tkwargs = wrapper(*args, **kwargs)

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g.add_amp_spline(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, spline_type)

    def add_phase_spline(self, *args, **kwargs):

        (x0, y0, c1, c2, c3,  ninterps, length, spline_type), tkwargs = wrapper(*args, **kwargs)

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g.add_phase_spline(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, spline_type)

    def add_orbit_information(self, *args, **kwargs):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
            links,
            sc_r, 
            sc_e,
            armlength
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g.add_orbit_information(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in,
            armlength
        )

    def add_tdi_config(self, *args, **kwargs):
        (
            unit_starts,
            unit_lengths,
            tdi_base_link,
            tdi_link_combinations,
            tdi_signs_in,
            channels,
            num_units,
            num_channels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t unit_starts_in = unit_starts
        cdef size_t unit_lengths_in = unit_lengths
        cdef size_t tdi_base_link_in = tdi_base_link
        cdef size_t tdi_link_combinations_in = tdi_link_combinations
        cdef size_t tdi_signs_in_in = tdi_signs_in
        cdef size_t channels_in = channels
        
        self.g.add_tdi_config(
            <int *>unit_starts_in, 
            <int *>unit_lengths_in, 
            <int *>tdi_base_link_in, 
            <int *>tdi_link_combinations_in, 
            <double *>tdi_signs_in_in, 
            <int *>channels_in, 
            num_units, 
            num_channels
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def check_x(self):
        self.g.check_x()

    def __reduce__(self):
        return (rebuild_td_spline, (self.amp_spline_ptr, self.phase_spline_ptr))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_td_spline_buffer_size(N)

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_length,
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phi_ref,
            params, t_arr,
            N, num_sub, n_params, nchannels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t buffer_in = buffer
        cdef size_t tdi_channels_arr_in = tdi_channels_arr
        cdef size_t tdi_amp_in = tdi_amp
        cdef size_t tdi_phase_in = tdi_phase
        cdef size_t phi_ref_in = phi_ref
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr

        self.g.run_wave_tdi(
            <void *>buffer_in, buffer_length,
            <cmplx *> tdi_channels_arr_in, 
        <double *> tdi_amp_in, <double *> tdi_phase_in,
        <double*> phi_ref_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params, nchannels)


def rebuild_td_spline(amp_spline_ptr, phase_spline_ptr):
    c = pyTDSplineTDIWaveform(amp_spline_ptr, phase_spline_ptr)
    return c



cdef class pyFDSplineTDIWaveform:
    cdef FDSplineTDIWaveformWrap *g

    def __cinit__(self):
        self.g = new FDSplineTDIWaveformWrap()

    
    def add_amp_spline(self, *args, **kwargs):

        (x0, y0, c1, c2, c3,  ninterps, length, spline_type), tkwargs = wrapper(*args, **kwargs)

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g.add_amp_spline(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, spline_type)

    def add_freq_spline(self, *args, **kwargs):

        (x0, y0, c1, c2, c3,  ninterps, length, spline_type), tkwargs = wrapper(*args, **kwargs)

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g.add_freq_spline(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, spline_type)

    def add_orbit_information(self, *args, **kwargs):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
            links,
            sc_r, 
            sc_e,
            armlength
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g.add_orbit_information(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in,
            armlength
        )

    def add_tdi_config(self, *args, **kwargs):
        (
            unit_starts,
            unit_lengths,
            tdi_base_link,
            tdi_link_combinations,
            tdi_signs_in,
            channels,
            num_units,
            num_channels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t unit_starts_in = unit_starts
        cdef size_t unit_lengths_in = unit_lengths
        cdef size_t tdi_base_link_in = tdi_base_link
        cdef size_t tdi_link_combinations_in = tdi_link_combinations
        cdef size_t tdi_signs_in_in = tdi_signs_in
        cdef size_t channels_in = channels
        
        self.g.add_tdi_config(
            <int *>unit_starts_in, 
            <int *>unit_lengths_in, 
            <int *>tdi_base_link_in, 
            <int *>tdi_link_combinations_in, 
            <double *>tdi_signs_in_in, 
            <int *>channels_in, 
            num_units, 
            num_channels
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild_fd_spline, (self.amp_spline_ptr, self.freq_spline_ptr, self.phase_ref_ptr))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_fd_spline_buffer_size(N)

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_length,
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phi_ref,
            params, t_arr,
            N, num_sub, n_params, nchannels
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t buffer_in = buffer
        cdef size_t tdi_channels_arr_in = tdi_channels_arr
        cdef size_t tdi_amp_in = tdi_amp
        cdef size_t tdi_phase_in = tdi_phase
        cdef size_t phi_ref_in = phi_ref
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr

        self.g.run_wave_tdi(
            <void *>buffer_in, buffer_length,
            <cmplx *> tdi_channels_arr_in, 
        <double *> tdi_amp_in, <double *> tdi_phase_in,
        <double*> phi_ref_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params, nchannels)

def rebuild_fd_spline(amp_spline_ptr, freq_spline_ptr, phase_ref_ptr):
    c = pyFDSplineTDIWaveform(amp_spline_ptr, freq_spline_ptr, phase_ref_ptr)
    return c