import time
import logging
from logging import info, getLogger
import argparse
import sys

from MLAO_correction import ml_estimate, polynomial_estimate

from pathlib import Path
from graph_results import graph
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
        self.model = 0
        self.bias_modes = False
        self.bias_magnitude = False
        # self.centerpixel = 64
        self.centerrange = 15
        self.path = ".//results"
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def Experiment(method, params):
    def graph_exp(jsonfilelist):
        for jsonfile, prefix in jsonfilelist:
            jsonpath = Path(jsonfile)
            folder, parent, name = jsonpath.parent, jsonpath.parent.name, jsonpath.name
            print(folder)
            graph(prefix, str(folder), [name])

    if method == "quadratic":
        if params.bias_modes and params.bias_magnitude:

            jsonfilelist = polynomial_estimate(params.bias_modes, params.bias_magnitude, params)
            graph_exp(jsonfilelist)
        else:
            model = ModelWrapper(params.model)
            jsonfilelist = polynomial_estimate(model.bias_modes, model.bias_magnitude, params)

    elif method == "mlao":
        jsonfilelist = ml_estimate(params)
        graph_exp(jsonfilelist)

    elif method == "comparison":
        jsonfilelist = ml_estimate(params)
        graph_exp(jsonfilelist)
        model = ModelWrapper(params.model)
        jsonfilelist = polynomial_estimate(model.bias_modes, model.bias_magnitude, params)
        graph_exp(jsonfilelist)

    else:
        log.warning("Experiment Parameters not recognised")


def run_experiments(experiments):
    "Run repeatable series of experiments for testing different ML models and comparing to conventional correction"
    params = mlao_parameters(modelno=experiments.model)
    t0 = time.time()
    log.info("----EXPERIMENTS START----")
    # calibrate: run quadratic correction for calibrate iterations
    if experiments.quadratic:
        params.update(
            scan_modes=[0],
            iter=experiments.quadratic,
            use_bias_only=True,
            experiment_name=f"_quadratic_M{experiments.model}",
        )
        Experiment("quadratic", params)
        log.info(f"----QUADRATIC ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    # Stability: scan 0 use bias_only 15 iterations
    if experiments.stability:
        params.update(scan=0, iter=18, experiment_name=f"stability_M{experiments.model}")
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
            iter=15,
            magnitude=experiments.scan_all,
            use_bias_only=False,
            experiment_name=f"_scan_all_M{experiments.model}",
        )

        Experiment(experiments.method, params)
        log.info(f"----SCAN ALL ESTIMATION COMPLETE T={(time.time()-t0)/60:0.1f} min----")
    log.info(f"----EXPERIMENTS COMPLETE T={(time.time()-t0)/60:0.1f} min----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #    "--dummy", help="runs in dummy mode without calling doptical/grpc", action="store_true"
    # )
    parser.add_argument(
        "-stability", help="Run 18 iterations of correction using system aberrations", action="store_true"
    )
    parser.add_argument("-single_abb", help="Plus/minus of specified aberration", type=int, default=0)
    parser.add_argument("-single_abb_mag", help="Plus/minus of specified aberration", type=int, default=3)
    parser.add_argument(
        "-quadratic", help="Run conventional (quadratic) correction over all modes", type=int, default=0
    )
    parser.add_argument("-scan_bias", help="", type=float, default=0)
    parser.add_argument("-scan_all", help="", type=float, default=0)

    parser.add_argument(
        "-method",
        help="Specifies experiment type FOR ALL EXPERIMENTS:'mlao' or 'comparison' or 'quadratic'",
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

    if args.log == "info":
        log.setLevel(20)
    elif args.log == "debug":
        log.setLevel(10)
    elif args.log == "warning":
        log.setLevel(30)
    elif args.log == "error":
        log.setLevel(40)

    run_experiments(args)