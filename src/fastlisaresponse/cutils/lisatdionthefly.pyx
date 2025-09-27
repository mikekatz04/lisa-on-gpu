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