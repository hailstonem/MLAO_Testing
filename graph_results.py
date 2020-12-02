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


correction = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.1256637061435919,
        0.2513274122871838,
        0.5026548245743672,
        4.44e-16,
        0.3769911184307757,
        -0.12566370614359146,
        0.18849555921538785,
        -0.06283185307179551,
        4.44e-16,
        0.06283185307179595,
        -0.12566370614359146,
        -0.06283185307179551,
        4.44e-16,
        -0.06283185307179551,
        4.44e-16,
        -0.06283185307179551,
        4.44e-16,
        -0.06283185307179551,
    ]
)

# 28_09_20
correction = np.array(
    [
        0.0,
        0.0,
        0.0,
        0,
        4.440892098500626e-16,
        0.5026548245743672,
        0.18849555921538785,
        -0.06283185307179551,
        0.1256637061435919,
        4.440892098500626e-16,
        0.31415926535897976,
        0.1256637061435919,
        -0.06283185307179551,
        0.18849555921538785,
        -0.12566370614359146,
        4.440892098500626e-16,
        -0.06283185307179551,
        4.440892098500626e-16,
        -0.06283185307179551,
        4.440892098500626e-16,
        0.06283185307179595,
        -0.06283185307179551,
    ]
)

# 121020
correction = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.3769911184307757,
        0.2513274122871838,
        0.1256637061435919,
        -0.06283185307179551,
        0.06283185307179595,
        0.06283185307179595,
        0.1256637061435919,
        0.06283185307179595,
        4.440892098500626e-16,
        0.06283185307179595,
        4.440892098500626e-16,
        4.440892098500626e-16,
        -0.06283185307179551,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
    ]
)
correction = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.2513274122871838,
        0.06283185307179595,
        0.06283185307179595,
        0.06283185307179595,
        0.06283185307179595,
        0.06283185307179595,
        0.1256637061435919,
        -2.243994752564138,
        4.440892098500626e-16,
        0.06283185307179595,
        4.440892098500626e-16,
        4.440892098500626e-16,
        -0.06283185307179551,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
        4.440892098500626e-16,
    ]
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


def graph(prefix, folder, filelist):
    # print(filelist)
    plt.figure()
    applied = []
    estimated = []
    indexes = []
    first = True
    # single_file = False

    it_dict = Iterations_dict(folder, filelist)
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
    plt.savefig(folder + "//" + prefix + "_convergence.png")

    plt.cla()
    # brightness plot
    plt.plot(d["Brightness"])
    plt.xlabel("Iteration")
    plt.ylabel("Brightness")
    plt.savefig(folder + "//" + prefix + "_brightness.png")
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

