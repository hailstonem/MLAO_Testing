import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS"] = "1"
import tensorflow as tf

# from tensorflow.keras import backend as K
import numpy as np


import grpc

from PySide2.QtWidgets import QApplication

DEBUG=False
if not DEBUG:
    from doptical.api.scanner_pb2_grpc import ScannerStub
    from doptical.api.scanner_pb2 import Empty, ZernikeModes, ScannerRange, ScannerPixelRange

    def capture_image(scanner):
        scanner.StartScan(Empty())

        images_available = False
        while not images_available:
            images_length = scanner.GetScanImagesLength(Empty()).length

            if images_length > 0:
                images_available = True

        images = scanner.GetScanImages(Empty()).images

        # assert(len(images) == 1)

        image = images[0]

        return_image = np.array(image.data).reshape(image.height,image.width)
        return return_image

def ML_estimate():
    """Runs ML estimation over a series of modes, printing the estimate of each mode and it's actual value."""

    model = tf.keras.models.load_model(
        "./models/"
        + "28CS-L45-90-45m-N2-MSE-xAR5CCFJ2-e3000-5000r25-175-HN-NB-rlu-A45C67S11DREAL37R21-IM15-TrN3-CA025-ScycLR-Mpl-adW-b25s6-1p200g-mpk5L-p05-92"
        + "_savedmodel.h5",
        compile=False,
    )

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
    for mode in return_modes:

        """ ###Not sure if necessary to preset aberration or not###
        ZM = ZernikeModes(modes=[mode],amplitudes=[1])
        scanner.SetSLMZernikeModes(modes)
        """

        ###TODO: Either use below code to make list of list of aberrations, or perhaps use list of ZernikeModes objects? depending on GetAOCalibrationStack
        start_aberrations = np.zeros((19))
        start_aberrations[mode - 3] = 1
        list_of_aberrations_lists = make_betas_polytope(start_aberrations, modes, 22, steps=[1])
        ###

        ###TODO: Rename Me, potentially edit signature####
        pixel_size = 0.1  ### TODO: set to ~2x Nyquist please
        image_dim = (128, 128)  # set as appropriate
        # image_dim = (500, 500)  # set as appropriate
        # stack = scanner.GetAOCalibrationStack(list_of_aberrations_lists, pixel_size, image_dim)
        
        # Set up scan
        scanner.SetScanPixelRange(ScannerPixelRange(x=image_dim[1],y=image_dim[0]))

        # Get stack of images
        aberration_modes = [int(i) for i in range(len(list_of_aberrations_lists))]
        stack = np.zeros((image_dim[0], image_dim[1], len(list_of_aberrations_lists)))
        for i_image, aberration in enumerate(list_of_aberrations_lists):
            print(aberration)

            ZM = ZernikeModes(modes=aberration_modes,amplitudes=aberration)
            scanner.SetSLMZernikeModes(ZM)
            image = capture_image(scanner)
            stack[:,:,i_image] = image
            time.sleep(1)
        

        # format for CNN
        stack = stack[np.newaxis, :, :, :]
        stack[stack < 0] = 0

        rot90 = False  # if it doesn't work for asymmetric modes but does for symmetric ones, set to True to check if caused by rotation problem
        if rot90:
            stack = np.rot90(stack, axes=[1, 2])

        pred = list(
            model.predict((stack.astype("float") - stack.mean()) / stack.std())[0]
        )  # list of estimated modes
        #pred=[0]*len(return_modes)

        print("Mode " + str(mode) + " Applied = " + str(1))
        print(
            "Mode "
            + str(mode)
            + " Estimate = "
            + str(pred[return_modes.index(mode)])
        )


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


ML_estimate()
