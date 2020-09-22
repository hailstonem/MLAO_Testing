# MLAO_Testing

[Now outdated]
 
Run from somewhere that can import doptical.api, but also loads ./models

I am assuming that the best way to run the ML_Testing is from a separate virtual environment, (dependencies are in requirements.txt). The version of tensorflow I have specified is cpu-only, so no hardware should be required. I am assuming that it will be straightforward to do this, but am happy to discuss other options when I am back. It may be that the tensorflow-cpu dependencies are not problematic, and can be installed in the same environment as doptical.

I didn't get the full signature for the doptical image streaming, so it may require some tweaking to get it working when Matthew completes that side of things.

I get the following output when I ran it in testing:

- 2020-09-04 18:23:01.766188: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4708 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
- Mode 4 Applied = 1
- Mode 4 Estimate = 0
- Mode 5 Applied = 1
- Mode 5 Estimate = 0
- Mode 6 Applied = 1
- Mode 6 Estimate = 0
- etc...

Obviously this is a very simple example, but if we can get this working then that would be a great step, and can move on to iterative correction when I'm back (or Qi, if you fancy editing it, it should be simple enough to do.)
