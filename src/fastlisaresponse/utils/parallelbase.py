from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class FastLISAResponseParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        force_backend_in = ('fastlisaresponse', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)

    @staticmethod
    def GPU_RECOMMENDED_WITH_JAX() -> list[str]:
        """Same as GPU_RECOMMENDED() but with the JAX backend appended.

        The JAX backend is a host-side pure-Python path (no compiled
        wheel needed); we list it after the GPU options so the default
        "first available" pick stays GPU when both are present.

        Returns bare platform tags (``cuda13x`` / ``cpu`` / ...); the
        consuming :meth:`supported_backends` prepends the
        ``fastlisaresponse_`` namespace to match the names registered
        in ``Globals().backends_manager``.
        """
        return ["cuda13x", "cuda12x", "cuda11x", "cpu", "jax"]
