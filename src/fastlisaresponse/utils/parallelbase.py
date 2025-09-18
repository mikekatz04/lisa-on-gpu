from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class FastLISAResponseParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        force_backend_in = ('fastlisaresponse', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)
