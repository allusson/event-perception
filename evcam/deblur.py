"""Event-based deblurring. Stub for now: EDI and deconvolution come later.

The inputs these will need are already available from `evcam.io.load_aedat4`: each APS
frame comes with `exposure_begin_t` and `exposure_end_t` on the same clock as the events,
so the events that fall inside a frame's exposure window can be selected directly.
"""


def edi_deblur(*args, **kwargs):
    """Event-based Double Integral deblurring (Pan et al., CVPR 2019). Not implemented yet."""
    raise NotImplementedError("EDI deblurring is not implemented yet")


def deconvolution_deblur(*args, **kwargs):
    """Deconvolution-based deblurring. Not implemented yet."""
    raise NotImplementedError("deconvolution deblurring is not implemented yet")
