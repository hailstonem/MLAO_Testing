import numpy as np


class Empty:
    def __init__(self):
        pass


class ScannerPixelRange:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ZernikeModes:
    def __init__(self, modes, amplitudes):
        pass


class ScannerStub:
    ### dummy ScannerStub for testing###
    def __init__(self, channel):
        pass

    def SetSLMZernikeModes(self, modes):
        pass

    def SetScanPixelRange(self, PixelRange):
        self.x = PixelRange.x
        self.y = PixelRange.y

    def GetAOCalibrationStack(self, list_of_aberrations_lists, pixel_size, image_dim):
        return np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))

    def StartScan(self, e):
        pass

    def GetScanImages(self, e):
        T = namedtuple("T", ["images"])
        return T([np.zeros((self.y, self.x))])

    def GetScanImagesLength(self, e):
        T = namedtuple("T", ["length"])
        return T(1)


def capture_image(scanner):
    return np.zeros(
        (scanner.y, scanner.x), dtype="uint16"
    )  # np.random.randint(0,65500,(scanner.y,scanner.x))#np.random.randint(0,65500,(scanner.y,scanner.x))
