import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Detector.hpp":
    ctypedef void* Orbits 'Orbits'

cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'
    
cdef extern from "TDIonTheFly.hh":
    cdef cppclass GBTDIonTheFlyWrap "GBTDIonTheFly":
        GBTDIonTheFlyWrap(Orbits *orbits, double T_) except+
        void dealloc() except+
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N) except+
        int get_gb_buffer_size(int N) except+
    cdef cppclass CubicSplineWrap "CubicSpline":
        CubicSplineWrap(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double dx, int N_, int spline_type_) except+
        void eval(double *y_new, double *x_new, int N) except+
        double eval_single(double x_new) except+
        void dealloc() except+

    cdef cppclass TDSplineTDIWaveformWrap "TDSplineTDIWaveform":
        TDSplineTDIWaveformWrap(Orbits *orbits_, CubicSplineWrap *amp_spline_, CubicSplineWrap *phase_spline_) except+
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N) except+
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
            N,
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

        self.g.run_wave_tdi(<void *>buffer_in, buffer_size, <double *>Xamp_in, <double *>Xphase_in, <double *>Yamp_in, <double *>Yphase_in, <double *>Zamp_in, <double *>Zphase_in, <double *>params_in, <double *>t_arr_in, N)

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


cdef class pyCubicSplineWrap:
    cdef CubicSplineWrap *g
    cdef uintptr_t x0_ptr
    cdef uintptr_t y0_ptr
    cdef uintptr_t c1_ptr
    cdef uintptr_t c2_ptr
    cdef uintptr_t c3_ptr
    cdef double dx
    cdef int N
    cdef int spline_type


    def __cinit__(self, x0, y0, c1, c2, c3, dx, N, spline_type):
        self.x0_ptr = x0
        self.y0_ptr = y0
        self.c1_ptr = c1
        self.c2_ptr = c2
        self.c3_ptr = c3
        self.dx = dx
        self.N = N
        self.spline_type = spline_type

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g = new CubicSplineWrap(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, dx, N, spline_type)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild_cublic_spline, (self.x0_ptr, self.y0_prt, self.c1_ptr, self.c2_ptr, self.c3_ptr, self.dx, self.spline_type))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def eval_single(self, x_new):
        return self.g.eval_single(x_new)

    def eval(self, *args, **kwargs):
        (y_new, x_new, N), tkwargs = wrapper(*args, **kwargs)
        cdef size_t x_new_in = x_new
        cdef size_t y_new_in = y_new
        self.g.eval(<double*> y_new_in, <double*> x_new_in, N)

def rebuild_cublic_spline(x0_ptr, y0_ptr, c1_ptr, c2_ptr, c3_ptr, dx, spline_type):
    c = pyCubicSplineWrap(x0_ptr, y0_ptr, c1_ptr, c2_ptr, c3_ptr, dx, spline_type)
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
        
        self.g = new TDSplineTDIWaveformWrap(<Orbits*>orbits_in, <CubicSplineWrap*>amp_spline_in, <CubicSplineWrap*>phase_spline_in)

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
            N,
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
        self.g.run_wave_tdi(<void *>buffer_in, buffer_size, <double *>Xamp_in, <double *>Xphase_in, <double *>Yamp_in, <double *>Yphase_in, <double *>Zamp_in, <double *>Zphase_in, <double *>params_in, <double *>t_arr_in, N)


def rebuild_td_spline(orbits_ptr, amp_spline_ptr, phase_spline_ptr):
    c = pyTDSplineTDIWaveform(orbits_ptr, amp_spline_ptr, phase_spline_ptr)
    return c