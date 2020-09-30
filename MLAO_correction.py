import os
import time
import json
import argparse

import numpy as np
import tifffile

import grpc
from PySide2.QtWidgets import QApplication

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS"] = "1"
import tensorflow as tf


def ml_estimate(iterations, scan, params):

    """Runs ML estimation over a series of modes, printing the estimate of each mode and it's actual value. 
    params should specify correct_bias_only load_abb and save_abb"""

    rnd = non_colliding_prefix("./results")

    model = ModelWrapper()
    bias_modes, return_modes = model.bias_modes, model.return_modes

    channel = grpc.insecure_channel("localhost:50051")
    scanner = ScannerStub(channel)

    if scan == -1:
        scan_modes = return_modes
    else:
        scan_modes = [scan]
    print(scan_modes)

    # loop over each mode and test to see if network estimates it
    for mode in scan_modes:

        # set initial aberration
        start_aberrations = np.zeros((max(return_modes) + 1))
        if params.load_abb:
            start_aberrations = load_start_abb("./start_abb.json", start_aberrations)

        start_aberrations[mode] += 1

        for it in range(iterations + 1):

            list_of_aberrations_lists = make_bias_polytope(start_aberrations, bias_modes, 22, steps=[1])

            # Set up scan
            image_dim = (128, 128)  # set as appropriate
            scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1], y=image_dim[0]))

            # Get stack of images
            aberration_modes = [int(i) for i in range(len(start_aberrations))]
            print(aberration_modes)
            stack = np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))

            # randomize image collection to minimise effects of bleaching
            shuffled_order = np.arange(len(list_of_aberrations_lists))
            np.random.shuffle(shuffled_order)

            # Get stack of images
            for i_image in shuffled_order:
                aberration = list_of_aberrations_lists[i_image]
                print([np.round(a, 1) for a in aberration])

                ZM = ZernikeModes(bias_modes=aberration_modes, amplitudes=aberration)
                scanner.SetSLMZernikeModes(ZM)
                time.sleep(1)
                image = capture_image(scanner)
                stack[:, :, i_image] = image

            # format for CNN
            stack = -stack[np.newaxis, 2:, 2:, :]  # Image is inverted (also clip flyback)
            # stack[stack < 0] = 0  ### is this necessary given we're working with floats?

            rot90 = False  # if it doesn't work for asymmetric modes but does for symmetric ones, set to True to check if caused by rotation problem
            # get prediction
            pred = model.predict(stack, rot90)

            print("Mode " + str(mode) + " Applied")
            if mode in return_modes:
                print("Mode " + str(mode) + " Estimate = " + str(pred[return_modes.index(mode)]))

            # save to json and tif
            jsonfile = "./results/%03d_%s_coefficients.json" % (rnd, mode,)
            if not params.correct_bias_only:
                coeff_to_json(jsonfile, start_aberrations, return_modes, pred, it + 1)
            else:
                coeff_to_json(
                    jsonfile,
                    start_aberrations,
                    bias_modes,
                    pred[[n for m, n in enumerate(return_modes) if m in bias_modes]],
                    it + 1,
                )

            tifname = "./results/%03d_%s_before.tif" % (rnd, mode)
            save_tif(tifname, stack[0, :, :, 0].astype("float32"))  # /stack[0, :, :, 0].max())

            # apply correction for next iteration
            if not params.correct_bias_only:
                start_aberrations[return_modes] = start_aberrations[return_modes] - pred
            else:
                start_aberrations[bias_modes] = start_aberrations[bias_modes] - np.array(pred)[bias_modes]

            # collect corrected image
            list_of_aberrations_lists = make_bias_polytope(start_aberrations, bias_modes, 22, steps=[])
            ZM = ZernikeModes(bias_modes=aberration_modes, amplitudes=list_of_aberrations_lists[0])
            scanner.SetSLMZernikeModes(ZM)
            image = capture_image(scanner)
            tifname = "./results/%03d_%s_after.tif" % (rnd, mode)
            save_tif(tifname, image[2:, 2:].astype("float32") / -1)

            if params.save_abb:
                with open("./start_abb.json", "w") as cofile:
                    data = dict(zip(return_modes, [float(p) for p in start_aberrations[return_modes]]))
                    json.dump(data, cofile, indent=1)


class ModelWrapper:
    """Stores model specific parameters and applied preprocessing before prediction"""

    def __init__(self):

        print("loading model")
        self.model = tf.keras.models.load_model(
            "./models/"
            + "28CS-L45-90-45m-N2-MSE-xAR5CCFJ2-e3000-5000r25-175-HN-NB-rlu-A45C67S11DREAL37R21-IM15-TrN3-CA025-ScycLR-Mpl-adW-b25s6-1p200g-mpk5L-p05-92"
            + "_savedmodel.h5",
            compile=False,
        )
        print("model_loaded")

        self.bias_modes = [4, 5, 6, 7, 10]  ### Bias modes
        self.return_modes = [
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
        ]

    def predict(self, stack, rot90=False):
        def rotate(stack, rot90):
            return np.rot90(stack, k=rot90, axes=[1, 2])

        if rot90:
            stack = rotate(stack, rot90)

        pred = list(
            self.model.predict(
                (
                    (stack.astype("float") - stack.mean())
                    / max(stack.astype("float").std(), 10e-20)  # prevent div/0
                )
            )[0]
        )
        return pred


def append_to_json(filename, new_data):
    if os.path.isfile(filename):
        with open(filename, "r") as cofile:
            data = json.load(cofile)
    else:
        data = {}
    with open(filename, "w") as cofile:
        data = {**data, **new_data}
        json.dump(data, cofile, indent=1)


def coeff_to_json(filename, start_aberrations, return_modes, pred, iterations):
    coeffs = dict()
    coeffs[str(iterations)] = {
        "Applied": dict(zip(return_modes, [float(p) for p in start_aberrations[return_modes]],)),
        "Estimated": dict(zip(return_modes, [float(p) for p in pred])),
    }
    append_to_json(filename, coeffs)


def save_tif(filename, data):

    tifffile.imsave(filename, data, append=True)


def non_colliding_prefix(path):
    indexes = set(np.arange(0, 999))
    used_index = set([int(x.split("_")[0]) for x in os.listdir(path) if len(x.split("_")[0]) == 3])
    return np.random.choice(list(indexes - used_index))


def load_start_abb(filename, abb):
    if os.path.isfile(filename):
        with open(filename, "r") as cofile:
            data = json.load(cofile)
    for k, v in data.items():
        abb[int(k)] = float(v)
    return abb


def make_bias_polytope(start_aberrations, offset_axes, nk, steps=(1)):
    """Return list of list of zernike amplitudes ('betas') for generating cross-polytope pattern of psfs
    """
    # beta (diffraction-limited), N_beta = cpsf.czern.nk
    beta = np.zeros(nk, dtype=np.float32)
    beta[:] = start_aberrations[:]
    # beta[0] = 1.0
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dummy", help="runs in dummy mode without calling doptical/grpc", action="store_true"
    )
    parser.add_argument("-iter", help="specifies number of iterations for correction", type=int, default=0)
    parser.add_argument(
        "-scan",
        help="values>3 apply 1 radian of the specified mode, 0-3 apply no aberration, -1 scans through each mode and corrects",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--correct_bias_only",
        help="values>3 apply 1 radian of the specified mode, 0-3 apply no aberration, -1 scans through each mode and corrects",
        action="store_true",
    )
    parser.add_argument(
        "--load_abb", help="if true, load intial aberration from json", action="store_true",
    )
    parser.add_argument(
        "--save_abb", help="if true, load intial aberration from json", action="store_true",
    )
    args = parser.parse_args()

    if args.dummy:
        from dummy_scanner import (
            ScannerStub,
            Empty,
            ZernikeModes,
            ScannerPixelRange,
            capture_image,
        )
    else:
        from doptical.api.scanner_pb2_grpc import ScannerStub
        from doptical.api.scanner_pb2 import Empty, ZernikeModes, ScannerRange, ScannerPixelRange

        def capture_image(scanner):
            scanner.StartScan(Empty())
            t0 = time.time()
            images_available = False
            while not images_available:
                time.sleep(1)
                images_length = scanner.GetScanImagesLength(Empty()).length

                if images_length > 0:
                    time.sleep(1)
                    images_available = True

            images = scanner.GetScanImages(Empty()).images

            # assert(len(images) == 1)

            image = images[0]

            return_image = np.array(image.data).reshape(image.height, image.width)
            scanner.StopScan(Empty())

            return return_image

    ml_estimate(args.iter, args.scan, args)
