# MLAO_Testing

Interface for collecting datasets and running simple tests and comparisons of ML/conventional methods.

Readme in progress

## Setup

Clone from github:

Using anaconda prompt:
```conda env create -f environment.yaml```



If using dm, dowload dmlib, instructions here [https://github.com/jacopoantonello/dmlib], and install dmlib. I did by running
``` setup.py develop```
in each of dmlib, zernike, and devwraps folders, but in principle installing directly from the install.bat files should be possible.

Run dm control:
```app.py```
--sim flag can be used for testing on systems with no dm.


## Running an experiment

Run doptical if it is not already running.

See "How to Run and Experiment" for further details, or use _experiment -h_ to see supported args
