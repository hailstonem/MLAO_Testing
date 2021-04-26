##
## Simple DM control example
## 
## - uses DM control object directly

import numpy as np
from h5py import File
import argparse

from dmlib.core import SquareRoot, FakeDM
from dmlib.calibration import RegLSCalib
from dmlib.control import ZernikeControl

# Scanner URI for MCAO microscope
SCANNER_URI = "10.200.20.36:50051"

# Parse arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sim', action='store_true')
args = parser.parse_args()

if args.sim:
    dm = FakeDM()
else:
    from devwraps.bmc import BMC
    dm = BMC() 
    dm.set_transform(SquareRoot())

devices = dm.get_devices()
dm.open(devices[0])

# Load DM calibration
calib_file = './data/calib.h5'
with File(calib_file, 'r') as f:
    calib = RegLSCalib.load_h5py(f, lazy_cart_grid=True)

# Create zernike control object
zcontrol = ZernikeControl(dm, calib)

# Get zernike array of correct length to write to DM control
z = np.zeros(shape=zcontrol.ndof)

# Write new data to control
z_new = np.random.uniform(-0.9,0.9,size=z.shape)
zcontrol.write(z_new)

# Set up client connection to scanner etc...
