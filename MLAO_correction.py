import os
import time
import json
import argparse
from logging import getLogger
import numpy as np
from numpy.polynomial.polynomial import Polynomial, polyval

import tifffile
from calibration import get_calibration
import grpc
from PySide2.QtWidgets import QApplication

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS"] = "1"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
log = getLogger("mlao_log")
try:
    from doptical.api.scanner_pb2_grpc import ScannerStub
    from doptical.api.scanner_pb2 import (
        Empty,
        ZernikeModes,
        ScannerRange,
        ScannerPixelRange,
        ImageStackID,
    )
except ImportError:
    log.warning("Running in Dummy mode")
    from dummy_scanner import (
        ScannerStub,
        Empty,
        ZernikeModes,
        ScannerPixelRange,
        ImageStackID,
    )


def Intensity_Metric(imagegen, centerpixel=50, centerrange=15):
    centermin = centerpixel - centerrange
    centermax = centerpixel + centerrange
    intensities = [np.sum(im[centermin:centermax, centermin:centermax]) for im in imagegen]
    return np.array(intensities)


def optimisation(coeffarray, metric, degree_fitting=2):

    new_series_fit = Polynomial.fit(coeffarray, metric, degree_fitting)

    new_coeffarray = np.linspace(np.max(coeffarray), np.min(coeffarray), 101)

    new_series = polyval(new_coeffarray, new_series_fit.convert().coef)

    maxcoeff = new_coeffarray[np.argmax(new_series)]
    log.debug(metric)

    return maxcoeff


def set_slm_and_capture_image(scanner, image_dim, aberration, aberration_modes, repeats):
    image = np.zeros(image_dim)
    for _ in range(repeats):
        log.debug([np.round(a, 1) for a in aberration])

        ZM = ZernikeModes(modes=aberration_modes, amplitudes=aberration)
        scanner.SetSLMZernikeModes(ZM)

        time.sleep(1.5)
        image += capture_image(scanner)[2:, 2:]
    image = -image  # Image is inverted (also clip flyback)
    image = image[:, ::-1]  # new correct flip?
    return image


class AberrationHistory:
    def __init__(self, initial_aberration):
        self.aberration = [np.array(initial_aberration)]
        self.prediction = []

    def update(self, aberration=None, prediction=None):
        log.debug(f"Aberration: {aberration},Predicted {prediction}")
        if aberration is not None and prediction is not None:
            self.aberration.append(aberration.copy())
            self.prediction.append(prediction.copy())
        elif aberration is not None:
            self.aberration.append(aberration.copy())
            self.prediction.append(self.aberration[-1] - self.aberration[-2])
        elif prediction is not None:
            self.prediction.append(prediction.copy())
            self.aberration.append(self.aberration[-1] - prediction)


def polynomial_estimate(bias_modes, return_modes, bias_magnitude, params):
    """Performs correction based on 3n polynomial fitting. Same API as ml_estimate, but also requires bias_modes parameter"""
    iterations = params.iter

    rnd = time_prefix("./results")
    folder = params.path + "/" + time.strftime("%y%m%" + "d") + params.experiment_name
    jsonfilelist = []
    if not os.path.exists(folder):
        os.mkdir(folder)

    channel = grpc.insecure_channel("localhost:50051")
    scanner = ScannerStub(channel)

    if params.scan == -1:
        scan_modes = return_modes
    elif params.scan == -2:
        scan_modes = bias_modes
    else:
        scan_modes = [params.scan]
    log.debug(f"scan modes: {scan_modes}")

    for mode in scan_modes:

        start_aberrations = np.zeros((max(bias_modes) + 1))
        jsonfile = folder + "/%03d_%s_coefficients.json" % (rnd, mode,)
        tifname = folder + "/%03d_%s_iterations.tif" % (rnd, mode)

        if params.load_abb:
            start_aberrations = load_start_abb("./start_abb.json", start_aberrations)
            log.debug("abberation loaded")

        if mode:
            start_aberrations[mode] += params.magnitude
        tracked_aberrations = start_aberrations
        aberration_modes = [int(i) for i in range(len(tracked_aberrations))]
        abb_his = AberrationHistory(start_aberrations)

        # Set up scan
        image_dim = (128, 128)  # set as appropriate
        scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1] + 2, y=image_dim[0] + 2))

        for it in range(iterations + 1):
            log.info(f"it {it} coefficients:{[np.round(a, 1) for a in tracked_aberrations]}")

            for bn, bias in enumerate(bias_modes):
                # Get lists of biases and aberrations
                list_of_aberrations_lists = make_bias_polytope(
                    tracked_aberrations, [bias], max(bias_modes) + 1, steps=[bias_magnitude]
                )

                # randomize image collection to minimise effects of bleaching
                shuffled_order = np.arange(len(list_of_aberrations_lists))
                if params.shuffle:
                    np.random.shuffle(shuffled_order)

                # Get stack of images
                stack = np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))
                for i_image in shuffled_order:
                    aberration = list_of_aberrations_lists[i_image]

                    image = set_slm_and_capture_image(
                        scanner, image_dim, aberration, aberration_modes, params.repeats
                    )

                    temptifname = folder + "/%03d_%s_%s_temp_.tif" % (rnd, mode, it)
                    save_tif(temptifname, image)

                    stack[:, :, i_image] = image

                # format
                stack = np.rollaxis(stack, 2, 0)

                # calculate metric
                intensities = Intensity_Metric(stack, stack.shape[1] // 2, params.centerrange)
                coeffarray = [x[bias] for x in list_of_aberrations_lists]
                optimal = optimisation(coeffarray, intensities, degree_fitting=2)
                log.debug(f"Mode {bias} Fit: {optimal}")

                # save tif and brightness from image before correction
                if bn == 0:
                    log.debug("saving tif")
                    save_tif(tifname, stack[0, :, :].astype("float32"))
                    brightness = np.mean(stack[0])
                # use this as starting point for next correction
                tracked_aberrations[bias] = optimal
            abb_his.update(aberration=tracked_aberrations)
            # log.info(abb_his.aberration[it])
            # log.info(abb_his.prediction[it][bias_modes])
            # save to json and tif
            coeff_to_json(
                jsonfile,
                abb_his.aberration[it],
                bias_modes,
                abb_his.prediction[it][bias_modes],
                it + 1,
                brightness=brightness,
                name="quadratic",
            )
            # if not params.correct_bias_only:
            #    coeff_to_json(jsonfile, start_aberrations, return_modes, pred, it + 1)
            # else:
            if it == iterations:

                list_of_aberrations_lists = make_bias_polytope(
                    tracked_aberrations, bias_modes, max(bias_modes) + 1, steps=[]
                )
                image = set_slm_and_capture_image(
                    scanner, image_dim, list_of_aberrations_lists[0], aberration_modes, 1
                )
                save_tif(tifname, image.astype("float32"))
                coeff_to_json(
                    jsonfile,
                    tuple(tracked_aberrations),
                    bias_modes,
                    np.zeros_like(tracked_aberrations)[bias_modes],
                    it + 2,
                    brightness=np.mean(image),
                    name="quadratic",
                )
        jsonfilelist.append((jsonfile, "%03d_%s" % (rnd, mode)))
    return jsonfilelist


def ml_estimate(params):

    iterations, scan = params.iter, params.scan
    """Runs ML estimation over a series of modes, printing the estimate of each mode and it's actual value. 
    params should specify correct_bias_only load_abb and save_abb"""

    rnd = time_prefix("./results")
    folder = "./results/" + time.strftime("%y%m%" + "d") + params.experiment_name
    if not os.path.exists(folder):
        os.mkdir(folder)
    jsonfilelist = []
    model = ModelWrapper(params.model)
    bias_magnitude, bias_modes, return_modes = model.bias_magnitude, model.bias_modes, model.return_modes

    # calibration should be from applied modes
    calibration = get_calibration([7])
    log.debug(calibration)
    channel = grpc.insecure_channel("localhost:50051")
    scanner = ScannerStub(channel)

    if scan == -1:
        scan_modes = return_modes
    elif scan == -2:
        scan_modes = bias_modes
    else:
        scan_modes = [scan]
    log.debug(f"scan modes: {scan_modes}")

    if len(params.disable_mode) > 0:
        log.debug(params.disable_mode)
        modifiable_modes = [r for r in return_modes if r not in [int(m) for m in params.disable_mode]]
        modifiable_mode_indexes = [en for en, m in enumerate(return_modes) if m in modifiable_modes]

    elif not params.correct_bias_only:
        modifiable_modes = return_modes
        modifiable_mode_indexes = [en for en, m in enumerate(return_modes) if m in modifiable_modes]
        # acc_pred / (it + 1)
    else:
        modifiable_modes = bias_modes
        modifiable_mode_indexes = [en for en, m in enumerate(return_modes) if m in modifiable_modes]
    log.debug("mm" + str(modifiable_modes))

    # loop over each mode and test to see if network estimates it
    for mode in scan_modes:

        # set initial aberration
        start_aberrations = np.zeros((max(return_modes) + 1))
        if params.load_abb:
            start_aberrations = load_start_abb("./start_abb.json", start_aberrations)
            log.debug("abberation loaded")

        if mode:
            start_aberrations[mode] += params.magnitude

        acc_pred = np.zeros(len(return_modes))
        old_brightness = 0

        for it in range(iterations + 1):
            log.info(f"it {it} coefficients:{[np.round(a, 1) for a in start_aberrations]}")
            # Set up scan
            image_dim = (128, 128)  # set as appropriate
            scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1] + 2, y=image_dim[0] + 2))

            # Get lists of biases and aberrations
            list_of_aberrations_lists = make_bias_polytope(
                start_aberrations, bias_modes, max(return_modes) + 1, steps=[bias_magnitude]
            )
            aberration_modes = [int(i) for i in range(len(start_aberrations))]
            stack = np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))

            # randomize image collection to minimise effects of bleaching
            shuffled_order = np.arange(len(list_of_aberrations_lists))
            if params.shuffle:
                np.random.shuffle(shuffled_order)

            # Get stack of images
            for i_image in shuffled_order:
                aberration = list_of_aberrations_lists[i_image]

                image = set_slm_and_capture_image(
                    scanner, image_dim, aberration, aberration_modes, params.repeats
                )

                temptifname = folder + "/%03d_%s_%s_temp_.tif" % (rnd, mode, it)
                save_tif(temptifname, image)

                if image is None:
                    raise RuntimeError("Image capture failed")
                stack[:, :, i_image] = image

            # format for CNN
            stack = stack[np.newaxis, :, :, :]  # Image is inverted (also clip flyback)
            # stack = stack[:, :, ::-1, :]  # new correct flip?

            """rot90 = False  # align rotation of image with network
            # save images
            tifffile.imsave(
                folder + "/%03d_%s_full_stack.tif" % (rnd, mode), np.rollaxis(stack.astype("float32"), 3, 1)
            )"""

            pred = [x / params.factor for x in model.predict(stack, split=False)]
            if params.use_calibration:
                pred = pred + 0.9 * calibration

            if mode in return_modes:
                log.info(f"Mode {mode} Applied; Estimate = {str(pred[return_modes.index(mode)])}")

            # save to json and tif
            jsonfile = f"{folder}/{rnd:03d}_{mode}_coefficients.json"
            jsonfilelist.append((jsonfile, "%03d_%s" % (rnd, mode)))
            # if not params.correct_bias_only:
            #    coeff_to_json(jsonfile, start_aberrations, return_modes, pred, it + 1)
            # else:
            coeff_to_json(
                jsonfile,
                tuple(start_aberrations),
                modifiable_modes,
                [pred[i] for i in modifiable_mode_indexes],
                it + 1,
                brightness=np.mean(stack[0, :, :, 0]),
                name="ml",
            )

            tifname = folder + "/%03d_%s_iterations.tif" % (rnd, mode)
            save_tif(tifname, stack[0, :, :, 0].astype("float32"))  # /stack[0, :, :, 0].max())

            acc_pred += pred

            # apply correction for next iteration

            start_aberrations[modifiable_modes] = (
                start_aberrations[modifiable_modes] - np.array(pred)[modifiable_mode_indexes]
            )

            # collect corrected image
            if it == iterations:

                list_of_aberrations_lists = make_bias_polytope(
                    start_aberrations, bias_modes, max(return_modes) + 1, steps=[]
                )
                image = set_slm_and_capture_image(
                    scanner, image_dim, list_of_aberrations_lists[0], aberration_modes, 1
                )
                save_tif(tifname, image[2:, 2::-1].astype("float32") / -1)

            brightness = np.sum(image)
            old_brightness = brightness.copy()
            if brightness < old_brightness:
                start_aberrations[modifiable_modes] = (
                    start_aberrations[modifiable_modes] + np.array(pred)[modifiable_mode_indexes]
                )

            if params.save_abb:
                with open("./start_abb.json", "w") as cofile:
                    data = dict(zip(return_modes, [float(p) for p in start_aberrations[return_modes]]))
                    json.dump(data, cofile, indent=1)
    return jsonfilelist


"""
def capture_image(scanner):
    # time.sleep(0.5)
    scanner.StartScan(Empty())
    # time.sleep(1)
    t0 = time.time()
    images_available = False
    while not images_available:
        time.sleep(0.1)
        images_length = scanner.GetScanImagesLength(Empty()).length

        if images_length > 0:
            time.sleep(0.1)
            images_available = True

    images = scanner.GetScanImages(Empty()).images

    # assert(len(images) == 1)

    image = images[0]

    return_image = np.array(image.data).reshape(image.height, image.width)
    scanner.StopScan(Empty())

    return return_image
"""


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
            raise RuntimeError("Image capture failed: Timeout on image capture")

        # retry delay
        time.sleep(retry_delay / 1000)

    image = images[0]

    return_image = np.array(image.data).reshape(image.height, image.width)
    scanner.StopScan(Empty())

    return return_image.astype("float32")


class ModelWrapper:
    """Stores model specific parameters and applied preprocessing before prediction"""

    def __init__(self, model_no=1):
        self.model = None
        self.bias_magnitude = 1
        self.model, self.subtract, self.return_modes = self.load_model(model_no)
        log.debug("model_loaded")
        self.bias_modes = [4, 5, 6, 7, 10]  ### Bias modes

    def load_model(self, model_no):
        log.debug("loading model")
        with open("./models/model_config.json", "r") as modelfile:
            model_dict = json.load(modelfile)
            log.debug(model_dict[str(model_no)])
        model_name = "./models/" + model_dict[str(model_no)][0] + "_savedmodel.h5"
        log.debug("model_name")
        model = tf.keras.models.load_model(model_name, compile=False,)
        subtract = model_dict[str(model_no)][1] == "S"
        self.return_modes = [int(x) for x in model_dict[str(model_no)][2]]
        if len(self.return_modes) == 0:
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
        self.bias_magnitude = [float(x) for x in model_dict[str(model_no)][3]]
        log.debug(self.bias_magnitude)
        if self.bias_magnitude == []:
            self.bias_magnitude = 1
        return model, subtract, self.return_modes

    def predict(self, stack, rot90=False, split=False):
        def rotate(stack, rot90):
            return np.rot90(stack, k=rot90, axes=[1, 2])

        if rot90:
            stack = rotate(stack, rot90)
        stack = (stack.astype("float") - stack.mean()) / max(
            stack.astype("float").std(), 10e-20
        )  # prevent div/0

        if self.subtract:
            stack = stack[:, :, :, 1:] - stack[:, :, :, 0:1]

        if split is False:
            pred = list(self.model.predict(stack)[0])
        else:
            pred = np.mean(
                [
                    self.model.predict(stack[:, 0 : stack.shape[1] * 3 // 4, 0 : stack.shape[2] * 3 // 4, :])[
                        0
                    ],
                    self.model.predict(stack[:, stack.shape[1] // 4 :, 0 : stack.shape[2] * 3 // 4, :])[0],
                    self.model.predict(stack[:, 0 : stack.shape[1] * 3 // 4, stack.shape[2] // 4 :, :])[0],
                    self.model.predict(stack[:, stack.shape[1] // 4 :, stack.shape[2] // 4 :, :])[0],
                ],
                axis=0,
                keepdims=False,
            )
        if len(pred) != len(self.return_modes):
            log.warning(
                f"Mismatch in returned modes: predicted:{len(pred)}, expected: {len(self.return_modes)}"
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


def coeff_to_json(filename, start_aberrations, return_modes, pred, iterations, brightness, name):
    coeffs = dict()
    coeffs[str(iterations)] = {
        "Applied": dict(zip(return_modes, [float(start_aberrations[p]) for p in return_modes],)),
        "Estimated": dict(zip(return_modes, [float(p) for p in pred])),
        "Brightness": float(brightness),
        "Type": str(name),
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
    parser.add_argument("-iter", help="specifies number of iterations for correction", type=int, default=0)
    parser.add_argument(
        "-scan",
        help="values>3 apply 1 radian of the specified mode, 0-3 apply no aberration, -1 scans through each mode and corrects",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--correct_bias_only", help="ignore model estimates other than bias modes", action="store_true",
    )
    parser.add_argument(
        "--load_abb", help="if true, load intial aberration from json", action="store_true",
    )
    parser.add_argument(
        "--save_abb", help="if true, load intial aberration from json", action="store_true",
    )
    parser.add_argument(
        "--disable_mode", help="select mode to disable", nargs="+", type=int, default=[],
    )
    parser.add_argument(
        "--use_calibration", help="whether to use calibration", action="store_true",
    )
    parser.add_argument(
        "--negative", help="whether to apply negative coefficient", action="store_true",
    )
    parser.add_argument(
        "--factor", help="divide prediction by number", type=int, default=1,
    )
    parser.add_argument("-repeats", help="apply averaging", type=int, default=1)
    parser.add_argument("-model", help="select model number", type=int, default=1)
    parser.add_argument(
        "-log", help="select logging level: info/debug/warning/error", type=str, default="info"
    )
    parser.add_argument("-path", help="output path", type=str, default=".//results")
    parser.add_argument("-experiment_name", help="add name for folder", type=str, default="")
    if args.log == "info":
        log.setLevel(20)
    elif args.log == "debug":
        log.setLevel(10)
    elif args.log == "warning":
        log.setLevel(30)
    elif args.log == "error":
        log.setLevel(40)

    args = parser.parse_args()

    if args.dummy:
        from dummy_scanner import (
            ScannerStub,
            Empty,
            ZernikeModes,
            ScannerPixelRange,
            ImageStackID,
        )
    ml_estimate(args)
