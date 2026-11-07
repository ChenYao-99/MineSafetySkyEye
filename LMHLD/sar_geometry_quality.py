"""Geometry diagnostics; uncertainty scenarios are never terrain observations."""
import numpy as np


def equivalent_ground_shift(line, pixel, delta_line, delta_pixel, spacing_x=30.,
                            spacing_y=-30., condition_max=20.):
    """Invert the local ground-to-image Jacobian, leaving singular cells unknown.

    The result is a first-order equivalent offset, not a measured location error.
    The signed y spacing is negative for a north-up raster.
    """
    line, pixel, dl, dp = map(lambda x: np.asarray(x, dtype=np.float64),
                              (line, pixel, delta_line, delta_pixel))
    if line.ndim != 2 or min(line.shape) < 3 or any(x.shape != line.shape for x in (pixel, dl, dp)):
        raise ValueError('Aligned 2D grids with at least three cells per axis required')
    if not np.isfinite([spacing_x, spacing_y, condition_max]).all() or spacing_x == 0 or spacing_y == 0 or condition_max < 1:
        raise ValueError('Invalid grid spacing or condition limit')
    ly, lx = np.gradient(line, spacing_y, spacing_x)
    py, px = np.gradient(pixel, spacing_y, spacing_x)
    det = lx*py - ly*px
    trace = lx*lx + ly*ly + px*px + py*py
    discriminant = np.sqrt(np.maximum(trace*trace - 4*det*det, 0))
    largest = (trace+discriminant)*.5
    smallest = np.divide(det*det, largest, out=np.zeros_like(det), where=largest>0)
    condition = np.sqrt(np.divide(largest, smallest, out=np.full_like(det, np.inf), where=smallest>0))
    good = np.isfinite(dl)&np.isfinite(dp)&np.isfinite(condition)&(condition<=condition_max)&(abs(det)>1e-12)
    good[[0,-1],:] = False
    good[:,[0,-1]] = False
    dx = np.divide(py*dl-ly*dp, det, out=np.full_like(det,np.nan), where=good)
    dy = np.divide(lx*dp-px*dl, det, out=np.full_like(det,np.nan), where=good)
    return dx, dy, good, condition


def historical_local_geometry(height, los_east, los_north, los_up, spacing_x=30., spacing_y=-30.):
    """Active slope risks from historic DSM and point-to-satellite ENU unit LOS.

    These local tests do not certify absence of passive or out-of-window shadows.
    """
    z, le, ln, lu = map(lambda x: np.asarray(x,dtype=np.float64), (height,los_east,los_north,los_up))
    if z.ndim!=2 or any(a.shape!=z.shape for a in (le,ln,lu)):
        raise ValueError('Height and LOS must share one 2D ground grid')
    zy,zx = np.gradient(z, spacing_y, spacing_x)
    horizontal = np.hypot(le,ln)
    slope = np.divide(zx*le+zy*ln, horizontal, out=np.full_like(z,np.nan), where=horizontal>0)
    # Positive range slope faces the sensor: height rises away from it.
    alpha = np.arctan(-slope)
    theta = np.arctan2(horizontal,lu)
    cosine = (-zx*le-zy*ln+lu)/np.sqrt(1+zx*zx+zy*zy)
    incidence = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    observed = np.isfinite(z)&np.isfinite(incidence)&(lu>0)&(horizontal>0)
    layover = observed&(alpha>=theta)
    backface = observed&(cosine<=0)
    return {'observed':observed,'local_incidence_deg':incidence,
            'active_layover_risk':layover,'local_backface_risk':backface}
