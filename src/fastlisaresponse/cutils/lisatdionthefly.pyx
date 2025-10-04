import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Detector.hpp":
    ctypedef void* Orbits 'Orbits'

cdef extern from "Interpolate.hh":
    ctypedef void* CubicSpline 'CubicSpline'

cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'
    
cdef extern from "TDIonTheFly.hh":
    cdef cppclass GBTDIonTheFlyWrap "GBTDIonTheFly":
        GBTDIonTheFlyWrap(Orbits *orbits, double T_) except+
        void dealloc() except+
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N, int num_sub, int n_params) except+
        int get_gb_buffer_size(int N) except+

    cdef cppclass TDSplineTDIWaveformWrap "TDSplineTDIWaveform":
        TDSplineTDIWaveformWrap(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_) except+
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N, int num_sub, int n_params) except+
        int get_td_spline_buffer_size(int N) except+
        void dealloc() except+

    cdef cppclass FDSplineTDIWaveformWrap "FDSplineTDIWaveform":
        FDSplineTDIWaveformWrap(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_) except+
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N, int num_sub, int n_params) except+
        int get_td_spline_buffer_size(int N) except+
        void dealloc() except+


cdef class pyGBTDIonTheFly:
    cdef Orbits *orbits
    cdef GBTDIonTheFlyWrap *g
    cdef double T
    cdef uintptr_t orbits_ptr

    def __cinit__(self, orbits_ptr, T):
        self.T = T 
        self.orbits_ptr = orbits_ptr
        cdef size_t orbits_in = orbits_ptr
        self.orbits = <Orbits *>orbits_in
        
        self.g = new GBTDIonTheFlyWrap(<Orbits *>orbits_in, self.T)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild, (self.orbits_ptr, self.T))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_size,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, t_arr,
            N, num_sub, n_params
        ), tkwargs = wrapper(*args, **kwargs)

        assert buffer_size == self.get_buffer_size(N)
        cdef size_t buffer_in = buffer
        cdef size_t Xamp_in = Xamp
        cdef size_t Xphase_in = Xphase
        cdef size_t Yamp_in = Yamp
        cdef size_t Yphase_in = Yphase
        cdef size_t Zamp_in = Zamp
        cdef size_t Zphase_in = Zphase
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr

        self.g.run_wave_tdi(<void *>buffer_in, buffer_size, <double *>Xamp_in, <double *>Xphase_in, <double *>Yamp_in, <double *>Yphase_in, <double *>Zamp_in, <double *>Zphase_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params)

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_gb_buffer_size(N)

def rebuild(dt,
    orbits_ptr,
    T,
):
    c = pyGBTDIonTheFly(
        orbits_ptr,
        T,
    )
    return c


cdef class pyTDSplineTDIWaveform:
    cdef TDSplineTDIWaveformWrap *g
    cdef uintptr_t orbits_ptr
    cdef uintptr_t amp_spline_ptr
    cdef uintptr_t phase_spline_ptr

    def __cinit__(self, orbits_ptr, amp_spline_ptr, phase_spline_ptr):
        self.orbits_ptr, self.amp_spline_ptr, self.phase_spline_ptr = orbits_ptr, amp_spline_ptr, phase_spline_ptr

        cdef size_t orbits_in = orbits_ptr
        cdef size_t amp_spline_in = amp_spline_ptr
        cdef size_t phase_spline_in = phase_spline_ptr
        
        self.g = new TDSplineTDIWaveformWrap(<Orbits*>orbits_in, <CubicSpline*>amp_spline_in, <CubicSpline*>phase_spline_in)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild_td_spline, (self.orbits_ptr, self.amp_spline_ptr, self.phase_spline_ptr))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_td_spline_buffer_size(N)

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_size,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, t_arr,
            N, num_sub, n_params
        ), tkwargs = wrapper(*args, **kwargs)

        assert buffer_size == self.get_buffer_size(N)
        cdef size_t buffer_in = buffer
        cdef size_t Xamp_in = Xamp
        cdef size_t Xphase_in = Xphase
        cdef size_t Yamp_in = Yamp
        cdef size_t Yphase_in = Yphase
        cdef size_t Zamp_in = Zamp
        cdef size_t Zphase_in = Zphase
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr
        self.g.run_wave_tdi(<void *>buffer_in, buffer_size, <double *>Xamp_in, <double *>Xphase_in, <double *>Yamp_in, <double *>Yphase_in, <double *>Zamp_in, <double *>Zphase_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params)


def rebuild_td_spline(orbits_ptr, amp_spline_ptr, phase_spline_ptr):
    c = pyTDSplineTDIWaveform(orbits_ptr, amp_spline_ptr, phase_spline_ptr)
    return c



cdef class pyFDSplineTDIWaveform:
    cdef FDSplineTDIWaveformWrap *g
    cdef uintptr_t orbits_ptr
    cdef uintptr_t amp_spline_ptr
    cdef uintptr_t freq_spline_ptr

    def __cinit__(self, orbits_ptr, amp_spline_ptr, freq_spline_ptr):
        self.orbits_ptr, self.amp_spline_ptr, self.freq_spline_ptr = orbits_ptr, amp_spline_ptr, freq_spline_ptr

        cdef size_t orbits_in = orbits_ptr
        cdef size_t amp_spline_in = amp_spline_ptr
        cdef size_t freq_spline_in = freq_spline_ptr
        
        self.g = new FDSplineTDIWaveformWrap(<Orbits*>orbits_in, <CubicSpline*>amp_spline_in, <CubicSpline*>freq_spline_in)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild_fd_spline, (self.orbits_ptr, self.amp_spline_ptr, self.freq_spline_ptr))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def get_buffer_size(self, N: int) -> int:
        return self.g.get_td_spline_buffer_size(N)

    def run_wave_tdi(self, *args, **kwargs):
        (
            buffer, buffer_size,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, t_arr,
            N, num_sub, n_params
        ), tkwargs = wrapper(*args, **kwargs)

        assert buffer_size == self.get_buffer_size(N)
        cdef size_t buffer_in = buffer
        cdef size_t Xamp_in = Xamp
        cdef size_t Xphase_in = Xphase
        cdef size_t Yamp_in = Yamp
        cdef size_t Yphase_in = Yphase
        cdef size_t Zamp_in = Zamp
        cdef size_t Zphase_in = Zphase
        cdef size_t params_in = params
        cdef size_t t_arr_in = t_arr
        self.g.run_wave_tdi(<void *>buffer_in, buffer_size, <double *>Xamp_in, <double *>Xphase_in, <double *>Yamp_in, <double *>Yphase_in, <double *>Zamp_in, <double *>Zphase_in, <double *>params_in, <double *>t_arr_in, N, num_sub, n_params)


def rebuild_fd_spline(orbits_ptr, amp_spline_ptr, freq_spline_ptr):
    c = pyFDSplineTDIWaveform(orbits_ptr, amp_spline_ptr, freq_spline_ptr)
    return c