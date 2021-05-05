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
in each of dmlib, zernike, and devwraps folders, but in principle installing directly as per the dmlib instructions should be possible.

## Running an experiment

Run doptical if it is not already running.

If using a dm, run the DM control _in the dmlib environment_ (e.g.```conda activate dmlib``` if dmlib is installed as per instructions):
```python app.py```
--sim flag can be used for testing on systems with no dm.

Run ```python experiment.py``` _in the MLAO environment_

See "How to Run and Experiment" for further details, or use _experiment -h_ to see supported args
