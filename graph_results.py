import os
import sys
import json
import argparse
from collections import UserDict, OrderedDict
import numpy as np
import tifffile
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("AGG")
sys.path.append("..\\ML-Zernicke-estimation\\")
#  # if plotting phase

# from imagegen import make_betas_polytope


def make_bias_polytope(start_aberrations, offset_axes, nk, steps=(1)):
    """Return list of list of zernike amplitudes ('betas') for generating cross-polytope pattern of psfs
    """
    # beta (diffraction-limited), N_beta = cpsf.czern.nk
    beta = np.zeros(nk, dtype=np.float32)
    beta[:] = start_aberrations[:]
    # beta[0] = 1.0
    # add offsets to beta

    betas = []
    betas.append(beta)
    for axis in offset_axes:
        for step in steps:
            plus_offset = beta.copy()
            plus_offset[axis] += 1 * step
            betas.append(plus_offset)
        for step in steps:
            minus_offset = beta.copy()
            minus_offset[axis] -= 1 * step
            betas.append(minus_offset)

    return betas


def splitx(x):
    def check_int(string):
        if string.isdigit():
            return int(string)
        else:
            return 0

    def split(stg):
        return tuple([check_int(stg.split("_")[xk]) for xk in x])

    return split


def save_phase(folder, prefix, index, est_acc, en):
    from fourier import Fraunhofer

    max_order = 6
    nk = (max_order + 1) * (max_order + 2) // 2
    NA = 1.2
    nd = 1.22 * 500e-9 / (2 * NA)
    pixel_size = nd
    fh = Fraunhofer(
        wavelength=500e-9,
        NA=NA,
        N=128,
        pixel_size=pixel_size / 2,
        n_alpha=6,
        image=np.ones((24, 24)).astype("uint16"),
    )
    start_abb = np.zeros(nk)
    # start_abb[6] = 1
    start_abb[[int(i) - 1 for i in index]] = np.array(est_acc)
    # print(est_acc)
    tifffile.imsave(
        folder + "//" + prefix + "//%02d.tif" % en,
        ((fh.phase(make_bias_polytope(start_abb, [5], nk, [])[0])).astype("float32")),
    )


def keytoint(d):
    k, v = d
    return int(k)


class Iterations_dict(UserDict):
    def __init__(self, folder, filelist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if just one file, we load this, to iterate through
        if len(filelist) == 1:
            with open(folder + "\\" + filelist[0]) as openfile:
                d_file = json.load(openfile)
                self.data = OrderedDict(sorted(d_file.items(), key=keytoint))
        else:
            self.data = OrderedDict()
            filelist.sort(key=splitx((0, 2)))
            for en, f in enumerate(filelist):
                with open(folder + f) as openfile:
                    d = json.load(openfile)
                    if "Estimated" in d:
                        self.data[str(en)] = d
                    elif "1" in d:
                        self.data[str(en)] = d["1"]
                    else:
                        print("No matching dict found")


def main(prefix, folder):
    # make output path
    if not os.path.isdir(folder + "//" + prefix + "//"):
        os.mkdir(folder + "//" + prefix + "//")

    filelist = [f for f in os.listdir(folder) if (f.endswith("json") and f.startswith(prefix))]
    print(folder)
    graph(prefix, folder, filelist)


def graph_compare(prefix, folder_ml, filelist_ml, folder_c, filelist_c):

    plt.figure()

    ml_dict = Iterations_dict(folder_ml, filelist_ml)
    ml_brightness = [b["Brightness"] for b in ml_dict.values()]

    c_dict = Iterations_dict(folder_c, filelist_c)
    c_brightness = [b["Brightness"] for b in c_dict.values()]

    # c_brightness = np.zeros((len(c_dict),))
    # for k, v in c_dict["Brightness"]:
    #    c_brightness[int(k)] = v

    plt.plot(np.arange(len(ml_brightness)), ml_brightness)
    plt.plot(np.arange(len(c_brightness)), c_brightness)
    plt.legend(["ML", "Quadratic"])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Brightness")
    # plt.scatter(indexes,estimated)
    plt.tight_layout()
    plt.savefig(folder_ml + "//" + prefix + "_brightness_comparison.png")
    plt.cla()


def graph(prefix, folder, filelist):
    """Refactored version to work with Experiment output"""
    # print(filelist)
    plt.figure()
    applied = []
    estimated = []
    indexes = []
    first = True

    # single_file = False
    # deal with single file vs multi-file import
    it_dict = Iterations_dict(folder, filelist)

    brightness = np.zeros((len(it_dict),))
    applied = np.zeros((len(it_dict), len(it_dict["1"]["Applied"])))
    estimated = np.zeros((len(it_dict), len(it_dict["1"]["Estimated"])))
    for i, d in it_dict.items():
        zero_i = int(i) - 1
        modes = [str(int(k) + 1) for k, v in d["Applied"].items()]
        applied[zero_i] = [float(x) for x in d["Applied"].values()]
        estimated[zero_i] = [float(x) for x in d["Estimated"].values()]
        brightness[zero_i] = [float(x) for x in d["Estimated"].values()]

    convergence_plot(modes, applied[:-1], folder, prefix)


def convergence_plot(modes, est_array, folder, prefix):

    for a in np.moveaxis(est_array, 1, 0):
        plt.plot([str(x) for x in np.arange(len(a))], a, marker=np.random.choice(["D", "o", "d", "s", "p"]))

    plt.legend(
        modes, loc="upper right", bbox_to_anchor=(1.25, 1),
    )
    plt.xlabel("Iteration")
    plt.ylabel("Radians")
    plt.tight_layout()
    plt.savefig(f"{folder}//{prefix}_convergence.png")
    plt.cla()


def brightness_plot(bright_array, folder, prefix):

    plt.plot(np.arange(len(bright_array)), bright_array)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Brightness")
    # plt.scatter(indexes,estimated)
    plt.tight_layout()
    plt.savefig(folder + "//" + prefix + "_brightness.png")
    plt.cla()


"""
def graph(prefix, folder, filelist):
    # print(filelist)
    plt.figure()
    applied = []
    estimated = []
    indexes = []
    first = True

    # single_file = False

    it_dict = Iterations_dict(folder, filelist)

    brightness = np.zeros((len(it_dict),))
    for en, d in it_dict.items():

        if first == True:
            est_acc = np.zeros(len(d["Estimated"]), dtype="float32")
            index = [str(int(i) + 1) for i, c in d["Applied"].items()]
            # plots starting point
            plt.scatter(
                index, [c for i, c in d["Applied"].items()], marker="D",
            )
            # plots reference correction
            plt.scatter(
                index, -correction[[int(i) for i, c in d["Estimated"].items()]], marker="D",
            )
            first = False
        # check indexes are consistent
        assert index == [str(int(i) + 1) for i, c in d["Applied"].items()]

        # indexes.append([str(int(i) + 1) for i, c in d["Applied"].items()])
        applied.append([c for i, c in d["Applied"].items()])

        # estimated.append([c for i, c in d['Estimated'].items()])
        # accumulate estimated aberration
        est_acc += np.array([c for i, c in d["Estimated"].items()])
        estimated.append(est_acc.copy())
        plt.scatter(index, est_acc)
        if "Brightness" in d:
            brightness[int(en) - 1] = d["Brightness"]
        # save_phase(folder, prefix, index, est_acc, int(en))

        # save_phase(folder, prefix, index, applied[-1], int(en))

    # all-in-one plot
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    # print(applied[0])
    plt.legend(
        ["applied", "Conventional"] + [str(x) for x in range(len(it_dict))],
        loc="upper right",
        bbox_to_anchor=(1.35, 1),
    )
    plt.ylim([-2, 2])
    plt.xlabel("Noll index")
    # plt.scatter(indexes,estimated)
    plt.savefig(folder + "//" + prefix + "_estimates.png")

    est_array = np.array(estimated)
    plt.cla()

    # convergence plot
    for a in np.moveaxis(est_array, 1, 0):

        plt.plot(
            [str(x + 1) for x in np.arange(len(a))], a, marker=np.random.choice(["D", "o", "d", "s", "p"])
        )

    plt.legend(
        [str(int(i)) for i, c in d["Estimated"].items()], loc="upper right", bbox_to_anchor=(1.25, 1),
    )
    plt.xlabel("Iteration")
    plt.ylabel("Radians")
    plt.tight_layout()
    plt.savefig(folder + "//" + prefix + "_convergence.png")

    plt.cla()
    # brightness plot
    if np.sum(brightness) > 0:

        plt.plot(np.arange(len(brightness)) + 1, brightness)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Brightness")
        # plt.scatter(indexes,estimated)
        plt.tight_layout()
        plt.savefig(folder + "//" + prefix + "_brightness.png")
        plt.cla()
"""
"""
    plt.scatter([str(int(i) + 1) for i, c in d["Estimated"].items()], est_acc)
    plt.scatter(
        [str(int(i) + 1) for i, c in d["Estimated"].items()],
        -correction[[int(i) for i, c in d["Estimated"].items()]],
    )
    plt.legend(["MLAO", "Conventional"], loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.savefig(folder + "//" + prefix + "//_comparison.png")
    plt.ylabel("Radians")
    plt.xlabel("Noll coefficient (1-indexed)")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", help="common starting characters in filenames", type=str)
    parser.add_argument("-folder", help="specify folder", type=str, default=".//results//")
    parser.add_argument("-parent", help="specify parent", type=str, default=None)
    args = parser.parse_args()
    if args.parent is not None:
        for f in os.listdir(args.parent):
            main("1", args.parent + f + "\\")
    elif args.prefix is not None:
        main(args.prefix, args.folder)
    else:
        print("specify prefix")

