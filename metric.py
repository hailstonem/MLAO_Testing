from logging import getLogger
from dataclasses import dataclass


import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal.windows import tukey
from numpy.polynomial.polynomial import Polynomial, polyval

log = getLogger("mlao_log")


def optimisation(coeffarray, metric, degree_fitting=2):
    """Polynomial fit over coeffarray range, returns maximum coefficient"""
    new_series_fit = Polynomial.fit(coeffarray, metric, degree_fitting)

    new_coeffarray = np.linspace(np.max(coeffarray), np.min(coeffarray), 101)

    new_series = polyval(new_coeffarray, new_series_fit.convert().coef)

    maxcoeff = new_coeffarray[np.argmax(new_series)]
    log.debug(metric)

    return maxcoeff


class MetricInterface:
    """Interface to choose metric"""

    def __init__(self, metric, img, nm_pixel_size=(100, 100), *kwargs):
        self.kwargs = kwargs
        if metric == "region":
            self.metric = self.regionintensitymetric
        elif metric == "total_intensity":
            self.metric = np.sum
        elif metric == "max_intensity":
            self.metric = np.max
        elif "low_spatial_frequencies" in metric:
            if metric.split("_")[-1] != "frequencies":
                nm_pixel_size = metric.split("_")[-1]
            imginfo = ImageInfo(img.shape, nm_pixel_size)
            self.metric = LowSpatialFrequencies(imginfo=imginfo, pars={}).eval

    def eval(self, img):
        return self.metric(img, *self.kwargs)

    @staticmethod
    def regionintensitymetric(imagegen, centerpixel=50, centerrange=15):
        """Total intensity over centre region of each image"""
        centermin = centerpixel - centerrange
        centermax = centerpixel + centerrange
        intensities = [np.sum(im[centermin:centermax, centermin:centermax]) for im in imagegen]
        return np.array(intensities)


@dataclass
class ImageInfo:
    shape: tuple
    pixel_size_nm: tuple


# From J. Antonello
class LowSpatialFrequencies:
    @staticmethod
    def get_default_parameters():
        return {
            "fl_nm": -1.0,
            "fh_nm": -1.0,
            "tukey_alpha": -1.0,
        }

    @staticmethod
    def get_parameters_info():
        return {
            "fl_nm": (float, (None, None), "lower frequency bound for the ring in [1/nm]"),
            "fh_nm": (float, (None, None), "higher frequency bound for the ring in [1/nm]"),
            "tukey_alpha": (float, (None, None), "shape parameter of the Tukey window"),
        }

    def __init__(self, imginfo, pars, h5f=None):
        r"""

 

        Parameters
        ----------
        - `imginfo`: `ImageInfo`
        - `pars`: `dict`
        - `h5f`: HDF5 file to log debug data

 

        References
        ----------
        ..  [D2007] Delphine Débarre, Martin J. Booth, and Tony Wilson, "Image
            based adaptive optics through optimisation of low spatial
            frequencies," Opt. Express 15, 8176-8190 (2007)
            <https://doi.org/10.1364/OE.15.008176>`__.

 

        """

        if "fl_nm" not in pars or pars["fl_nm"] < 0.0:
            fl_nm = None
        else:
            fl_nm = pars["fl_nm"]

        if "fh_nm" not in pars or pars["fh_nm"] < 0.0:
            fh_nm = None
        else:
            fh_nm = pars["fh_nm"]

        if "tukey_alpha" not in pars or pars["tukey_alpha"] < 0.0:
            tukey_alpha = 0.25
        else:
            tukey_alpha = pars["tukey_alpha"]

        N1, N2 = imginfo.shape
        pixel_size_nm = np.asanyarray(imginfo.pix_size) * 1e9
        # TODO
        assert N1 % 2 == 0

        # make mask in Fourier space
        ff1 = np.arange(-N1 // 2, N1 // 2) / (N1 * pixel_size_nm[0])
        ff2 = np.arange(-N2 // 2, N2 // 2) / (N2 * pixel_size_nm[1])
        xx, yy = np.meshgrid(ff2, ff1)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        if fl_nm is None or fh_nm is None:
            # typically the object has low frequency content
            fl_nm = 0.1 * min(ff1.max(), ff2.max())
            # measurement (Poisson) noise dominates at high freqs
            fh_nm = 0.6 * min(ff1.max(), ff2.max())
        mask = (rr > fl_nm) * (rr < fh_nm)

        win = tukey(N1, tukey_alpha, True).reshape(-1, 1) * tukey(N2, tukey_alpha, True).reshape(1, -1)

        if h5f:
            addr = f"metric/{self.__class__.__name__}"
            h5f[addr + "/pixel_size_nm"] = pixel_size_nm
            h5f[addr + "/fl_nm"] = fl_nm
            h5f[addr + "/fh_nm"] = fh_nm
            h5f[addr + "/tukey_alpha"] = tukey_alpha
            h5f[addr + "/win"] = win
            h5f[addr + "/mask"] = mask
            h5f[addr + "/ff1"] = ff1
            h5f[addr + "/ff2"] = ff2

        M = fftshift(mask)

        self.win = win
        self.M = M

    def eval(self, img):
        Fimg = fftshift(img.astype(np.float) * self.win)
        F = fft2(Fimg)
        FM = np.abs(F * self.M)
        return FM.sum()

