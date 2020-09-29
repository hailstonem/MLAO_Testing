import os
import json
import matplotlib
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from fourier import Fraunhofer
from imagegen import make_betas_polytope


def split3(stg):
    return int(stg.split("_")[-2])


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

folder = ".//results//"
prefix = "334"
if not os.path.isdir(folder + "//" + prefix + "//"):
    os.mkdir(folder + "//" + prefix + "//")
filelist = [f for f in os.listdir(folder) if (f.endswith("json") and f.startswith(prefix))]
filelist.sort(key=split3)
print(filelist)
plt.figure()
applied = []
estimated = []
indexes = []
first = True
single_file=False
if len(filelist) == 1:
    with open(folder + filelist[0]) as openfile:
        d_file = json.load(openfile)
        single_file=True
        filelist=d_file.keys()
for en, f in enumerate(filelist):
    if single_file:
        d = d_file[str(en+1)]
    else:
        with open(folder + f) as openfile:
            d = json.load(openfile)

    if first == True:
        est_acc = np.array([c for i, c in d["Estimated"].items()])
        first = False
        plt.scatter(
            [str(int(i) + 1) for i, c in d["Applied"].items()],
            [c for i, c in d["Applied"].items()],
            marker="D",
        )
        plt.scatter(
            [str(int(i) + 1) for i, c in d["Estimated"].items()],
            -correction[[int(i) for i, c in d["Estimated"].items()]],
            marker="D",
        )

    indexes.append([str(int(i) + 1) for i, c in d["Applied"].items()])
    applied.append([c for i, c in d["Applied"].items()])

    # estimated.append([c for i, c in d['Estimated'].items()])
    est_acc += np.array([c for i, c in d["Estimated"].items()])
    estimated.append(est_acc.copy())
    plt.scatter([str(int(i) + 1) for i, c in d["Estimated"].items()], est_acc)

    max_order = 6
    nk = (max_order + 1) * (max_order + 2) // 2
    NA = 1.2
    nd = 1.22 * 500e-9 / (2 * NA)
    pixel_size = nd
    fh = Fraunhofer(
        wavelength=500e-9,
        NA=NA,
        N=60,
        pixel_size=pixel_size / 2,
        n_alpha=6,
        image=np.ones((40, 40)).astype("uint16"),
    )
    start_abb = np.zeros(nk - 3)
    start_abb[[int(i) - 3 for i, c in d["Applied"].items()]] = est_acc

    tifffile.imsave(
        folder + "//" + prefix + "//%02d.tif" % en,
        ((fh.phase(make_betas_polytope(start_abb, [5], nk, [])[0])).astype("float32")),
    )


plt.tight_layout(rect=[0, 0, 0.8, 1])
# print(applied[0])
plt.legend(
    ["applied", "Conventional", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    loc="upper right",
    bbox_to_anchor=(1.35, 1),
)
plt.ylim([-2, 2])
plt.xlabel("Noll index")
# plt.scatter(indexes,estimated)
plt.savefig(folder + "//" + prefix + "//_estimates.png")

est_array = np.array(estimated)

plt.cla()
for a in np.moveaxis(est_array, 1, 0):

    plt.plot(a, marker=np.random.choice(["D", "o", "d", "s", "p"]))
plt.legend(
    [str(int(i) + 1) for i, c in d["Estimated"].items()],
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
)
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.savefig(folder + "//" + prefix + "//_convergence.png")


plt.cla()
plt.scatter([str(int(i) + 1) for i, c in d["Estimated"].items()], est_acc)
plt.scatter(
    [str(int(i) + 1) for i, c in d["Estimated"].items()],
    -correction[[int(i) for i, c in d["Estimated"].items()]],
)
plt.legend(["MLAO", "Conventional"], loc="upper right", bbox_to_anchor=(1.25, 1))
plt.savefig(folder + "//" + prefix + "//_comparison.png")
plt.ylabel("Radians")
plt.xlabel("Noll coefficient (1-indexed)")

