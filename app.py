import sys
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSlot as Slot, pyqtSignal as Signal
from concurrent import futures
import time
import numpy as np
import queue

import dmlib.zpanel as zpanel
import grpc
import dm_pb2
import dm_pb2_grpc


URI = "[::]:50052"

class App(QApplication):
    def __init__(self, args):
        super().__init__([])

        # Set up RPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        self.server.add_insecure_port(URI)

        self.dm_servicer = DM_servicer()
        self.dm_servicer.set_zernike_modes.connect(self.handle_zernike)
        dm_pb2_grpc.add_DMServicer_to_server(self.dm_servicer, self.server)

        self.server.start()

        # Set up DM + gui
        dm_pars = {'calibration': './data/calib.h5'}
        self.dmwin = zpanel.new_zernike_window(self, args, dm_pars)
        self.dmwin.show()

    def start_server(self):
        # Create grpc server with threads
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))

        # Add servicer to handle DM requests
        self.DM_servicer = DM()
        dm_pb2_grpc.add_DMServicer_to_server(self.DM_servicer, self._server)

        # Set port and start server
        self.server.add_insecure_port(URI)
        self.server.start()

    def handle_zernike(self,request):
        # Get request data
        modes = list(request.modes)
        amplitudes = list(request.amplitudes)

        # Log
        print("Set DM: {} - {}".format(modes, amplitudes))

        # Format data correctly
        z = np.zeros(shape=self.dmwin.zcontrol.ndof)

        for i,mode in enumerate(modes):
            z[mode-1] = amplitudes[i]
        
        # Write to DM and update GUI
        self.dmwin.write_dm(z)
        self.dmwin.zpanel.z[:] = self.dmwin.zcontrol.u2z()
        self.dmwin.zpanel.update_gui_controls()
        self.dmwin.zpanel.update_phi_plot()        


class DM_servicer(dm_pb2_grpc.DMServicer, QObject):
    set_zernike_modes = Signal(object)

    def __init__(self):
        super().__init__()

    def SetDMZernikeModes(self, request, context):
        # Emit signal
        self.set_zernike_modes.emit(request)

        # Return blank response
        return dm_pb2.Empty()

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sim', action='store_true')
    zpanel.add_arguments(parser)
    args = parser.parse_args()

    # Set DM parameters (simulated if in sim mode)
    if args.sim:
        args.dm_name = 'simdm0'
    else:
        args.dm_name = None
        args.dm_driver = 'bmc'

    # Create app and launch
    app = App(args)
    sys.exit(app.exec_())