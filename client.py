import grpc

from dm_pb2_grpc import DMStub
from dm_pb2 import Empty, ZernikeModes

import numpy as np
import time

dm_channel = grpc.insecure_channel("localhost:50052")
dm = DMStub(dm_channel)

# Test
while True:
    for i in range(10):
        modes = [i+1]
        amp = [1]

        ZM = ZernikeModes(modes=modes,amplitudes=amp)
        dm.SetDMZernikeModes(ZM)
        time.sleep(0.5)