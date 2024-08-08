import numpy as np

try:
    import cupy as cp
    from pyresponse import get_response_wrap as get_response_wrap_gpu
    from pyresponse import get_tdi_delays_wrap as get_tdi_delays_wrap_gpu

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False


def get_overlap(sig1, sig2, phase_maximize=False, use_gpu=False):
    """Calculate the mismatch across TDI channels

    Calculates the overlap between two sets of TDI observables in the time
    domain. The overlap is complex allowing for the addition of overlap
    over all channels. It can be phase maximized as well.

    This function has GPU capabilities.

    Args:
        sig1 (list or xp.ndarray): TDI observables for first signal. Must be ``list`` of
            ``xp.ndarray`` or a single ``xp.ndarray``. Must have same length as ``sig2`` in terms
            of number of channels and length of the indivudal channels.
        sig2 (list or xp.ndarray): TDI observables for second signal. Must be ``list`` of
            ``xp.ndarray`` or a single ``xp.ndarray``. Must have same length as ``sig1`` in terms
            of number of channels and length of the individual channels.
        phase_maximize (bool, optional): If ``True``, maximize over the phase in the overlap.
            This is equivalent to getting the magnitude of the phasor that is the complex
            overlap. (Defaut: ``False``)
        use_gpu (bool, optional): If ``True``, use the GPU. This sets ``xp=cupy``. If ``False,
            use the CPU and set ``xp=numpy``.

    Returns:
        double: Overlap as a real value.

    """

    # choose right array library
    if use_gpu:
        xp = cp
    else:
        xp = np

    # check inputs
    if not isinstance(sig1, list):
        if not isinstance(sig1, xp.ndarray):
            raise ValueError("sig1 must be list of or single xp.ndarray.")

        elif sig1.ndim < 2:
            sig1 = [sig1]

    if not isinstance(sig2, list):
        if not isinstance(sig2, xp.ndarray):
            raise ValueError("sig1 must be list of or single xp.ndarray.")

        elif sig1.ndim < 2:
            sig2 = [sig2]

    assert len(sig1) == len(sig2)
    assert len(sig1[0]) == len(sig2[0])

    # complex overlap
    overlap = 0.0 + 1j * 0.0
    for sig1_i, sig2_i in zip(sig1, sig2):
        overlap_i = np.dot(np.fft.rfft(sig1_i).conj(), np.fft.rfft(sig2_i)) / np.sqrt(
            np.dot(np.fft.rfft(sig1_i).conj(), np.fft.rfft(sig1_i))
            * np.dot(np.fft.rfft(sig2_i).conj(), np.fft.rfft(sig2_i))
        )
        overlap += overlap_i

    overlap /= len(sig1)

    if phase_maximize:
        return np.abs(overlap)

    else:
        return overlap.real
