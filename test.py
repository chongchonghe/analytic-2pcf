#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test.py

Run
-------
python test.py

Author
------
Chong-Chong He (che1234@umd.edu)

"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from time import time
import csv
import tpcf

def testnumba():

    N = 3000
    print(f"Testing numba acceleration by running dr_cuboid on {N} particles.")
    d = np.random.random([N, 3])
    dpart = d[:100, :]  # compile the numba machine code
    rs = np.logspace(-3, 0, 20)
    tpcf.dr_cuboid(dpart, rs, 1, 1, 1)
    t1 = time()
    tpcf.dr_cuboid(d, rs, 1, 1, 1)
    dt = time() - t1
    print(f"Time elapsed: {dt} sec")
    print(f"If numba is installed and working properly, the time shown above "
          "should be under 0.05 seconds.\n")

def get_eagle_data(NDIM=3, choose="1"):

    data_dir = "mock-data"
    sim_name = {"1": "RefL0050N0752_Subhalo",
                "2": "RefL0100N1504_Subhalo"}[choose]
    fn_data = os.path.join(data_dir, f"eagle_data_{sim_name}.npy")
    fn_info = os.path.join(data_dir, f"eagle_data_{sim_name}.info")
    with open(fn_data, "rb") as f:
        data = np.load(f)
    with open(fn_info, "r") as f:
        d = csv.reader(f)
        for l in d:
            boxlen = float(l[1])
            break
    assert np.all(data <= boxlen) and np.all(data >= 0.)
    if NDIM == 2:
        return data[:2, :].T, (boxlen, boxlen)
    elif NDIM == 3:
        return data.T, (boxlen, boxlen, boxlen)

def plot_test_tpcf():
    """ Plot analytically calculated TPCF
    """

    sample0, dim = get_eagle_data()
    N = 10000
    sample1 = sample0[:N, :]
    print(f"Doing {N} data particles")
    a, b, c = dim
    # rbins = np.logspace(-1, np.log10(a/2.0000001), 20)
    rbins = np.logspace(-1, np.log10(a), 20)
    rbins2 = (rbins[1:] + rbins[:-1]) / 2 / a

    est = 'LS'
    xi = tpcf.tpcf_ana(sample1, rbins, shape='cuboid', bound=dim, est=est)

    plt.plot(rbins2, xi, 'k',)
    plt.gca().set(xscale='log', yscale='log',
           ylabel=r"$\xi(r)$",
           xlabel="Normalized radius $r$",
           )
    plt.savefig("test.png", dpi=300)
    print("test.png saved")
    plt.show()

if __name__ == "__main__":

    testnumba()
    plot_test_tpcf()
