import os
import time
from collections import namedtuple

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS"] = "1"
import tensorflow as tf

# from tensorflow.keras import backend as K
import numpy as np

import tifffile
import json

import grpc

from PySide2.QtWidgets import QApplication

import argparse

DEBUG = True
if not DEBUG:
    from doptical.api.scanner_pb2_grpc import ScannerStub
    from doptical.api.scanner_pb2 import Empty, ZernikeModes, ScannerRange, ScannerPixelRange

    def capture_image(scanner):
        scanner.StartScan(Empty())
        t0 = time.time()
        images_available = False
        while not images_available:
            time.sleep(2)
            images_length = scanner.GetScanImagesLength(Empty()).length

            if images_length > 0:
                time.sleep(2)
                images_available = True

        images = scanner.GetScanImages(Empty()).images

        # assert(len(images) == 1)

        image = images[0]

        return_image = np.array(image.data).reshape(image.height, image.width)
        scanner.StopScan(Empty())

        return return_image


else:

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


def ML_estimate(iterative_correct, scan):
    """Runs ML estimation over a series of modes, printing the estimate of each mode and it's actual value."""
    print('loading model')
    model = tf.keras.models.load_model(
        "./models/"
        + "28CS-L45-90-45m-N2-MSE-xAR5CCFJ2-e3000-5000r25-175-HN-NB-rlu-A45C67S11DREAL37R21-IM15-TrN3-CA025-ScycLR-Mpl-adW-b25s6-1p200g-mpk5L-p05-92"
        + "_savedmodel.h5",
        compile=False,
    )
    print('model_loaded')
    rnd = np.random.randint(0, 999)
    modes = [4, 5, 6, 7, 10]  ### Bias modes (specific to model)
    return_modes = [
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        15,
        16,
        21,
    ]  ### These modes are the ones the model returns (specific to model)

    channel = grpc.insecure_channel("localhost:50051")
    scanner = ScannerStub(channel)

    # loop over each estimated mode and test to see if network estimates it
    # corrections
    if scan == -1:
        scan_modes = return_modes
    else:
        scan_modes = [scan]#[np.random.choice(return_modes)]
    print(scan_modes)
    for mode in scan_modes:

        start_aberrations = np.zeros((19))
        start_aberrations[mode - 3] = 1

        for it in range(iterative_correct + 1):
            ###TODO: Either use below code to make list of list of aberrations, or perhaps use list of ZernikeModes objects? depending on GetAOCalibrationStack

            list_of_aberrations_lists = make_betas_polytope(start_aberrations, modes, 22, steps=[1])
            ###

            ###TODO: Rename Me, potentially edit signature####
            pixel_size = 0.1  ### TODO: set to ~2x Nyquist please
            image_dim = (128, 128)  # set as appropriate
            # image_dim = (500, 500)  # set as appropriate
            # stack = scanner.GetAOCalibrationStack(list_of_aberrations_lists, pixel_size, image_dim)

            # Set up scan
            scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1], y=image_dim[0]))

            # Get stack of images
            aberration_modes = [int(i) for i in range(len(list_of_aberrations_lists))]
            stack = np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))
            for i_image, aberration in enumerate(list_of_aberrations_lists):
                print(aberration)

                ZM = ZernikeModes(modes=aberration_modes, amplitudes=aberration)
                scanner.SetSLMZernikeModes(ZM)
                image = capture_image(scanner)
                stack[:, :, i_image] = image
                time.sleep(1)

            # format for CNN
            stack = -stack[np.newaxis, :, :, :] #IMAGE INPUT IS INVERTED!!!
            stack[stack < 0] = 0

            rot90 = False  # if it doesn't work for asymmetric modes but does for symmetric ones, set to True to check if caused by rotation problem
            if rot90:
                stack = np.rot90(stack, axes=[1, 2])

            pred = list(
                model.predict(
                    (
                        (stack.astype("float") - stack.mean())
                        / max(stack.astype("float").std(), 10e-20) #prevent div/0
                    )
                )[0]
            )  # list of estimated modes
            # pred=[0]*len(return_modes)

            with open("./results/%s_%s_%s_coefficients.json" % (rnd, mode, it + 1), "w") as cofile:
                coeffs = {
                    "Applied": dict(zip(return_modes, [float(p) for p in start_aberrations])),
                    "Estimated": dict(zip(return_modes, [float(p) for p in pred])),
                }
                json.dump(coeffs, cofile, indent=1)

            tifffile.imsave(
                "./results/%s_%s_before_%s.tif" % (rnd, mode, it + 1), stack[0, :, :, 0].astype('float32')/stack[0, :, :, 0].max()
            )  # rnd just there to make overwrites unlikely. #TODO: Replace with proper solution when we have a better idea of what we want to save

            start_aberrations = np.zeros((19))

            scanner.SetSLMZernikeModes(ZM)
            image = capture_image(scanner)

            tifffile.imsave(
                "./results/%s_%s_after_%s.tif" % (rnd, mode, it + 1), -image.astype('float32')/(image.min())
            )  # rnd just there to make overwrites unlikely. Replace with proper solution when we have a better idea of what we want to save

            print("Mode " + str(mode) + " Applied = " + str(1))
            print("Mode " + str(mode) + " Estimate = " + str(pred[return_modes.index(mode)]))

            start_aberrations = pred


def make_betas_polytope(start_aberrations, offset_axes, nk, steps=[1]):
    """Return list of list of zernike amplitudes ('betas') for generating cross-polytope pattern of psfs
    """
    # beta (diffraction-limited), N_beta = cpsf.czern.nk
    beta = np.zeros(nk, dtype=np.float32)
    beta[0] = 1.0
    # ignore 0-2 Noll: piston, tip, tilt, start at defocus
    beta[3 : 3 + len(start_aberrations)] = start_aberrations

    # add offsets to beta

    betas = []
    betas.append(beta)
    for axis in offset_axes:
        for step in steps:
            plus_offset = beta.copy()
            plus_offset[axis] += 1 * step
            betas.append(plus_offset)
        for step in steps:
            minus_offset = beta.copy()
            minus_offset[axis] -= 1 * step
            betas.append(minus_offset)

    return betas


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dummy", help="runs in dummy mode without calling doptical/grpc", action="store_true"
)
parser.add_argument("-iter", help="number of iterations", type=int)
parser.add_argument(
    "-scan",
    help="if true scans through modes and applies and estimates single mode aberration, otherwise corrects for a single aberration", type=int
)
args = parser.parse_args()
ML_estimate(args.iter, args.scan)

