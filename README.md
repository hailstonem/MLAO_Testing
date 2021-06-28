# MLAO_Testing

Interface for collecting datasets and running simple tests and comparisons of ML/conventional methods.

## Setup

This setup assumes that you have conda installed (e.g. via Anaconda) and git.

Clone this repo from github:

```git clone https://github.com/hailstonem/MLAO_Testing.git```

Using (ana)conda prompt, navigate to MLAO_testing directory, and create the environment:
```conda env create -f environment.yaml```

If using dm, dowload dmlib, instructions here [https://github.com/jacopoantonello/dmlib], and install dmlib. I did by running
``` setup.py develop```
in each of dmlib, zernike, and devwraps folders, but installing directly as per the dmlib instructions is possible, but be careful about which environment it is installed into.

## Running an experiment

Run doptical if it is not already running.

If using a dm, run the DM control _in the dmlib environment_ (e.g.```conda activate dmlib``` if dmlib is installed as per instructions):
```python app.py```
--sim flag can be used for testing on systems with no dm.

Run ```python experiment.py``` _in the MLAO environment_

See "How to Run and Experiment" for further details, or use _experiment -h_ to see supported args

## Development

### Testing Closed loop correction: 'synthetic microscope'

You have a couple of options for this.

By default, ```experiment.py``` can be run with ```--dummy```. This will default to using random noise unless it can find _imagegen/imagegen.py_ from ML_Zernike_Estimation. This can be installed in the parent folder, so that it is accessible from _../ML_Zernike_Estimation/_. Alternatively, the requisite folder can be placed directly in the MLAO_Testing folder.

### DM testing 
With dmlib installed, it is possible to test full interaction of a 'fake dm', and doptical. To do so, run doptical in one process with --dummy, and app.py (set up with None as dm), and finally run the experiment.py layer on top. Note that experiment.py will also need to specify the correct GRPC channels (usually "localhost:50051" and "localhost:50052".
