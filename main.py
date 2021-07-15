#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""main.py

The the tpcf.py. Generate an analytically calculated two-point correlation function

Run
-------
python main.py

"""

#-----------------------------------------------------------------------------
#    Author: Chong-Chong He (che1234@umd.edu)
#    Data: 2021-05-05
#-----------------------------------------------------------------------------

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from time import time
import csv
import src

def test_numba():
    from time import time
    print("Testing numba acceleration")
    d = np.random.random([10000, 3])
    dpart = d[:100, :]
    rs = np.logspace(-3, 0, 20)
    src.dr_cuboid(dpart, rs, 1, 1, 1)
    t1 = time()
    src.dr_cuboid(d, rs, 1, 1, 1)
    dt = time() - t1
    print(f"Time elapsed: {dt} sec")
    print(f"If numba is working properly, this time should be under 1 second, "
          "likely under 0.1 second.")


def get_data_eagle(NDIM=3, choose="1"):

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


def main():
    """ Plot analytically calculated TPCF 
    """

    sample0, dim = get_data_eagle()
    N = 1000
    sample1 = sample0[:N, :]
    print(f"Doing {N} data particles")
    a, b, c = dim
    rbins = np.logspace(-1, np.log10(a/2.0000001), 20)
    rbins2 = (rbins[1:] + rbins[:-1]) / 2 / a

    est = 'LS'
    xi = src.analytic_tpcf(sample1, rbins, shape='cuboid', bounds=dim, est=est)

    plt.plot(rbins2, xi, 'k',)
    plt.gca().set(xscale='log', yscale='log',
           ylabel=r"$\xi(r)$",
           xlabel="Normalized radius $r$",
           )
    plt.show()


if __name__ == "__main__":

    main()
