##
## Simple DM control example using GUI
##
## - uses DM gui to display/control DM

import argparse
import sys
import logging
import subprocess
import os
import time

import numpy as np

from qtpy.QtWidgets import QApplication

import grpc
from scanner_pb2_grpc import ScannerStub
from scanner_pb2 import Empty, ScannerRange, ScannerPixelRange, ImageStackID

from h5py import File
from dmlib.core import SquareRoot, FakeDM
from dmlib.calibration import RegLSCalib
from dmlib.control import ZernikeControl
import dmlib.zpanel as zpanel


from gui import Main


class DM_Interface:
    """Wrapper around dmlib"""

    def __init__(self, args):

        """ Start scanner gui if in simulated mode
        if sim:
            app_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "doptical", "doptical", "app.py"
                )
            )
            cmd = [sys.executable, app_path, "-d"]
            scanner_process = subprocess.Popen(cmd)
            SCANNER_URI = "localhost:50051"
        """
        # Set DM parameters (simulated if in sim mode)
        if args.sim:
            args.dm_name = "simdm0"
        else:
            args.dm_name = None
            args.dm_driver = "bmc"

        # Show DM GUI
        app = QApplication(sys.argv)
        dm_pars = {"calibration": "./data/calib.h5"}
        self.dmwin = zpanel.new_zernike_window(app, args, dm_pars)
        self.dmwin.show()

        # Kick off app
        sys.exit(app.exec_())

    def write(self, aberrations):
        # Get zernike array of correct length to write to DM control
        z = np.zeros(shape=self.dmwin.zcontrol.ndof)
        z[: len(aberrations)] = aberrations
        # Write new data to control
        self.dmwin.write_dm(z)
        # Get DM control from gui attribute
        self.dmwin.zpanel.z[:] = self.dmwin.zcontrol.u2z()
        self.dmwin.zpanel.update_gui_controls()
        self.dmwin.zpanel.update_phi_plot()

