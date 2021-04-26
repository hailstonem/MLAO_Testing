import 
from logging import getLogger
log = getLogger("mlao_log")
from numpy.polynomial.polynomial import Polynomial, polyval


def Intensity_Metric(imagegen, centerpixel=50, centerrange=15):
    """Total intensity over centre region of each image"""
    centermin = centerpixel - centerrange
    centermax = centerpixel + centerrange
    intensities = [np.sum(im[centermin:centermax, centermin:centermax]) for im in imagegen]
    return np.array(intensities)


def optimisation(coeffarray, metric, degree_fitting=2):
    """Polynomial fit over coeffarray range, returns maximum coefficient"""
    new_series_fit = Polynomial.fit(coeffarray, metric, degree_fitting)

    new_coeffarray = np.linspace(np.max(coeffarray), np.min(coeffarray), 101)

    new_series = polyval(new_coeffarray, new_series_fit.convert().coef)

    maxcoeff = new_coeffarray[np.argmax(new_series)]
    log.debug(metric)

    return maxcoeff

def make_bias_polytope(start_aberrations, offset_axes, nk, steps=(1)):
    """Return list of list of zernike amplitudes ('betas') for generating cross-polytope pattern of psfs
    """
    # beta (diffraction-limited), N_beta = cpsf.czern.nk
    beta = np.zeros(nk, dtype=np.float32)
    beta[:] = start_aberrations
    beta[0] = 0.01
    # add offsets to beta

    betas = []
    betas.append(tuple(beta))
    for axis in offset_axes:
        for step in steps:
            plus_offset = beta.copy()
            plus_offset[axis] += 1 * step
            betas.append(tuple(plus_offset))
        for step in steps:
            minus_offset = beta.copy()
            minus_offset[axis] -= 1 * step
            betas.append(tuple(minus_offset))

    return betas