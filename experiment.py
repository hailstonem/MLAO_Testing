import time
import winsound
import logging
from logging import info, getLogger
import argparse
import sys
import os
import numpy as np
from MLAO_correction import ml_estimate, collect_dataset  # polynomial_estimate,

from pathlib import Path
from graph_results import graph, graph_compare
from MLAO_correction import ModelWrapper


# standard tests:

# full
# single mode pos/neg
# calibrate?


class mlao_parameters:
    """Container for MLAO default args"""

    def __init__(self, **kwargs):
        self.dummy = False
        self.shuffle = False
        self.scan = 0
        self.iter = 1
        self.correct_bias_only = False
        self.load_abb = False
        self.save_abb = False
        self.disable_mode = []
        self.use_calibration = False
        self.negative = False
        self.factor = 1
        self.repeats = 1
        self.magnitude = 1.5
        self.modelno = 0
        self.bias_modes = False
        self.bias_magnitude = False
        # self.centerpixel = 64
        self.centerrange = 62  # 15
        self.path = ".//results"
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def Dataset(params, kind=None):

    # Do initial correction
    params.update(
        save_abb=True, scan=0, iter=3, use_bias_only=True, modelno=-1, experiment_name=f"_system_M-1",
    )
    Experiment("quadratic", params)
    # Collect dataset
    step = 0.5
    if kind == "large":
        applied_steps = np.concatenate(
            [np.linspace(-4, -2, 2 / step, endpoint=False), np.linspace(2 + step, 4, 2 / step)]
        )
    else:
        applied_steps = np.linspace(-2, 2, (2 * 2) / step + 1)

    params.update(load_abb=True, shuffle=True)
    collect_dataset(
        bias_modes=[3, 4, 5, 6, 7, 10],
        applied_modes=[4, 5, 6, 7, 10],
        applied_steps=applied_steps,
        bias_magnitudes=[1, 2],
        params=params,
    )


def Experiment(method, params):
    def graph_exp(jsonfilelist):
        for jsonfile, prefix in jsonfilelist:
            jsonpath = Path(jsonfile)
            folder, parent, name = jsonpath.parent, jsonpath.parent.name, jsonpath.name
            print(folder)
            graph(prefix, str(folder), [name])

    def graph_exp_compare(jsonfilelist_ml, jsonfilelist_c):
        for ml, c in zip(jsonfilelist_ml, jsonfilelist_c):
            print(ml, c)
            jsonfile_ml, prefix_ml = ml
            jsonfile_c, prefix_c = c
            jsonpath_ml, jsonpath_c = Path(jsonfile_ml), Path(jsonfile_c)

            folder_ml, name_ml = jsonpath_ml.parent, jsonpath_ml.name
            folder_c, name_c = jsonpath_c.parent, jsonpath_c.name
            print(folder_ml)

            graph(prefix_ml, str(folder_ml), [name_ml])
            graph(prefix_c, str(folder_c), [name_c])
            graph_compare(prefix_ml, str(folder_ml), [name_ml], str(folder_c), [name_c])

    if method in ["quadratic", "q"]:
        if not (params.bias_modes and params.bias_magnitude):
            if not params.modelno:
                log.warning(
                    "Possible missing Experiment Parameters: model, or bias_modes and bias_magnitude: falling back to model 0"
                )
            model = ModelWrapper(params.modelno)

        # jsonfilelist = polynomial_estimate(model.bias_modes, model.return_modes, model.bias_magnitude, params)
        jsonfilelist = ml_estimate(params, quadratic=True)
        graph_exp(jsonfilelist)

    elif method in ["mlao", "m", "ml"]:
        jsonfilelist = ml_estimate(params)
        graph_exp(jsonfilelist)

    elif method in ["comparison", "compare", "c"]:
        jsonfilelist_ml = ml_estimate(params)
        # graph_exp(jsonfilelist_ml)
        jsonfilelist_c = ml_estimate(params, quadratic=True)
        """
        model = ModelWrapper(params.modelno)
        jsonfilelist_c = polynomial_estimate(
            model.bias_modes, model.return_modes, model.bias_magnitude, params
        )"""
        # graph_exp(jsonfilelist_c)
        graph_exp_compare(jsonfilelist_ml, jsonfilelist_c)
    else:
        log.warning("Experiment Parameters not recognised: make sure 'method' is set correctly")


def run_experiments(experiments):
    "Run repeatable series of experiments for testing different ML models and comparing to conventional correction"
    params = mlao_parameters(modelno=experiments.model, correct_bias_only=experiments.correct_bias_only,)
    t0 = time.time()
    log.info("----EXPERIMENTS START----")
    # calibrate: run quadratic correction for calibrate iterations
    if experiments.system:
        params.update(
            scan=0,
            iter=experiments.system,
            use_bias_only=True,
            experiment_name=f"_system_M{experiments.model}",
        )
        Experiment("quadratic", params)
        log.info(f"----QUADRATIC ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Stability: scan 0 use bias_only 15 iterations
    if experiments.stability:
        params.update(scan=0, iter=18, experiment_name=f"_stability_M{experiments.model}")
        Experiment("mlao", params)
        log.info(f"----STABILITY ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Single Large Abb: pos and negative at specified magnitude
    if experiments.single_abb:
        params.update(
            scan=experiments.single_abb,
            iter=5,
            magnitude=experiments.single_abb_mag,
            use_bias_only=True,
            experiment_name=f"_mode_{experiments.single_abb}_{experiments.single_abb_mag}_M{experiments.model}",
        )
        Experiment(experiments.method, params)
        params.update(
            scan=experiments.single_abb, iter=5, magnitude=-experiments.single_abb_mag, use_bias_only=True
        )
        Experiment(experiments.method, params)
        log.info(f"----SINGLE ABERRATION ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Large Abb: use_bias_only scan through bias modes
    if experiments.scan_bias:
        params.update(
            scan=-2,
            iter=5,
            magnitude=experiments.scan_bias,
            use_bias_only=True,
            experiment_name=f"_scan_bias_M{experiments.model}",
        )

        Experiment(experiments.method, params)
        log.info(f"----SCAN BIAS ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Small aberration regime:
    # full range scan through modes
    if experiments.scan_all:
        params.update(
            scan=-1,
            iter=5,
            magnitude=experiments.scan_all,
            use_bias_only=False,
            experiment_name=f"_scan_all_M{experiments.model}",
        )
        log.info(f"----SCAN ALL ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Dataset collection regime:
    if experiments.dataset:
        params.update(
            scan=-1,
            iter=5,
            magnitude=experiments.scan_all,
            use_bias_only=False,
            experiment_name=f"_Dataset_R{experiments.dataset}",
        )

        Dataset(params, experiments.dataset)
        log.info(f"----DATASET COLLECTION COMPLETE T={(time.time()-t0)/60:0.1f} min----")

    log.info(f"----EXPERIMENTS COMPLETE T={(time.time()-t0)/60:0.1f} min----")

    def play_note(note, duration=300):
        notes = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
        scale = 440
        ratio = 1.05946
        winsound.Beep(int(scale * ratio ** (notes[note])), duration)
        time.sleep(0.1)

    if not experiments.no_beep:
        song = "E E F G G F E D C C"
        for note in song.split():
            play_note(note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #    "--dummy", help="runs in dummy mode without calling doptical/grpc", action="store_true"
    # )
    parser.add_argument(
        "-stability",
        help="Run 18 iterations of correction using system aberrations. Uses parameters from specified model",
        action="store_true",
    )
    parser.add_argument(
        "-single_abb",
        help="Plus/minus of specified aberration.  Also specify single_abb_mag",
        type=int,
        default=0,
    )
    parser.add_argument("-single_abb_mag", help="single_abb Aberration magnitude", type=float, default=3)
    parser.add_argument(
        "-system",
        help="Run correction over all modes for specified iterations with no applied aberration",
        type=int,
        default=0,
    )
    parser.add_argument("-scan_bias", help="", type=float, default=0)
    parser.add_argument("-scan_all", help="", type=float, default=0)
    parser.add_argument("-dataset", help="", type=str, default="")
    parser.add_argument("--no_beep", help="disable beeping", action="store_true")
    parser.add_argument(
        "--correct_bias_only", help="ignore model estimates other than bias modes", action="store_true",
    )
    parser.add_argument(
        "-method",
        help="Specifies experiment type FOR ALL EXPERIMENTS:'mlao'(default) or 'comparison' or 'quadratic' (short form: m,c,q)",
        type=str,
        default="mlao",
    )
    parser.add_argument("-model", help="select model number", type=int, default=1)
    parser.add_argument(
        "-log", help="select logging level: info/debug/warning/error", type=str, default="info"
    )
    parser.add_argument("-output_path", help="", type=str, default=".//results")
    args = parser.parse_args()
    log = getLogger("mlao_log")
    handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    log.addHandler(handler)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.log == "info":
        log.setLevel(20)
    elif args.log == "debug":
        log.setLevel(10)
    elif args.log == "warning":
        log.setLevel(30)
    elif args.log == "error":
        log.setLevel(40)

    run_experiments(args)
