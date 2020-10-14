import os
import time
import json
import argparse

import numpy as np
import tifffile
from calibration import get_calibration
import grpc
from PySide2.QtWidgets import QApplication

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS"] = "1"


def test_image_capture():

    """T"""

    rnd = time_prefix("./results")
    folder = "./results/" + time.strftime("%y%m%" + "d")
    if not os.path.exists(folder):
        os.mkdir(folder)

    channel = grpc.insecure_channel("localhost:50051")
    scanner = ScannerStub(channel)
    scan_modes = [4, 5, 6, 7, 8, 9, 10]

    for mode in scan_modes:
        for magnitude in [0, 0.5, 1, 1.5, 2]:

            # Set up scan
            image_dim = (128, 128)  # set as appropriate
            scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1], y=image_dim[0]))

            start_aberrations = np.zeros((max(scan_modes) + 1))
            start_aberrations[mode] = magnitude
            aberration_modes = [int(i) for i in range(len(start_aberrations))]
            print(aberration_modes)
            print([np.round(a, 1) for a in start_aberrations])

            ZM = ZernikeModes(modes=aberration_modes, amplitudes=start_aberrations)
            scanner.SetSLMZernikeModes(ZM)
            # if params.dummy:
            time.sleep(1)

            temptifname = folder + "/%03d_%s_test_.tif" % (rnd, mode)
            save_tif(temptifname, capture_image(scanner, timeout=5000, retry_delay=10))


def capture_image(scanner, timeout=5000, retry_delay=10):
    id = scanner.StartScan(Empty()).id
    t_start = time.time()

    # Set image id to search for
    req_id = ImageStackID()
    req_id.id = id

    images_found = False
    while not images_found:
        images = scanner.GetScanImages(req_id).images

        if len(images):
            images_found = True

        t_elapsed = time.time() - t_start

        # Timeout if no image found
        if t_elapsed > timeout / 1000:
            print("TIMEOUT ON IMAGE CAPTURE")
            return None

        # retry delay
        time.sleep(retry_delay / 1000)

    image = images[0]

    return_image = np.array(image.data).reshape(image.height, image.width)
    scanner.StopScan(Empty())

    return return_image.astype("float32")


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
        "Applied": dict(zip(return_modes, [float(start_aberrations[p]) for p in return_modes],)),
        "Estimated": dict(zip(return_modes, [float(p) for p in pred])),
    }
    append_to_json(filename, coeffs)


def save_tif(filename, data):

    tifffile.imsave(filename, data, append=True)


def non_colliding_prefix(path):
    indexes = set(np.arange(0, 999))
    used_index = set([int(x.split("_")[0]) for x in os.listdir(path) if len(x.split("_")[0]) == 3])
    return np.random.choice(list(indexes - used_index))


def time_prefix(path):

    # index = np.random.randint(0, 10)
    prefix = int(time.strftime("%H%M%S"))
    # used_index = set([int(x.split("_")[0]) for x in os.listdir(path) if len(x.split("_")[0]) == 3])
    return prefix


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dummy", help="runs in dummy mode without calling doptical/grpc", action="store_true"
    )
    args = parser.parse_args()

    if args.dummy:
        from dummy_scanner import (
            ScannerStub,
            Empty,
            ZernikeModes,
            ScannerPixelRange,
            ImageStackID,
        )
    else:
        from doptical.api.scanner_pb2_grpc import ScannerStub
        from doptical.api.scanner_pb2 import (
            Empty,
            ZernikeModes,
            ScannerRange,
            ScannerPixelRange,
            ImageStackID,
        )

    test_image_capture()
