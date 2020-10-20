import sys
from collections import namedtuple
import numpy as np

sys.path.append("..\\ML-Zernicke-estimation\\")
from fourier import Fraunhofer
from imagegen import make_betas_polytope


class Empty:
    def __init__(self):
        pass


class ScannerPixelRange:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ZernikeModes:
    def __init__(self, modes, amplitudes):
        self.modes = modes
        self.amplitudes = amplitudes


class ScannerStub:
    ### dummy ScannerStub for testing###
    def __init__(self, channel):
        self.max_order = 6
        self.nk = (self.max_order + 1) * (self.max_order + 2) // 2
        self.NA = 1.2
        self.nd = 1.22 * 500e-9 / (2 * self.NA)
        self.pixel_size = self.nd / 2
        self.fh = None
        self.aberrations = np.zeros(self.nk)
        self.scan = None
        self.x = 0
        self.y = 0

    def SetSLMZernikeModes(self, ZM):
        assert len(ZM.modes) == len(ZM.amplitudes)
        aberrations = np.zeros(self.nk)
        aberrations[ZM.modes] = ZM.amplitudes
        self.aberrations = aberrations

    def SetScanPixelRange(self, PixelRange):
        """Not properly implemented"""
        self.x = PixelRange.x
        self.y = PixelRange.y

        image = np.zeros((self.y, self.x)).astype("uint16")
        image[60:62, 60] = 200
        image[2, 2] = 200
        image[-2, -2] = 200
        image[2, -2] = 200
        image[-2, 2] = 200

        self.fh = Fraunhofer(
            wavelength=500e-9, NA=self.NA, N=128, pixel_size=self.pixel_size / 1.5, n_alpha=6, image=image,
        )

    def GetAOCalibrationStack(self, list_of_aberrations_lists, pixel_size, image_dim):
        return np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))

    def StartScan(self, e):

        psf = self.fh.psf(self.aberrations)
        self.scan = self.fh.incoherent(psf, True)

        T = namedtuple("T", ["id"])(np.random.randint(1000))
        return T

    def StopScan(self, e):
        pass

    def GetScanImages(self, id):
        T = namedtuple("T", ["images"])
        scan = self.scan
        self.scan = None
        return T([ImageWrapper(scan)])

    def GetScanImagesLength(self, e):
        T = namedtuple("T", ["length"])
        return T(1)


class ImageStackID:
    def __init__(self):
        self.id = 0


class ImageWrapper:
    def __init__(self, image):
        self.data = -image[:, ::-1]  # inverted
        self.height = image.shape[0]
        self.width = image.shape[1]
