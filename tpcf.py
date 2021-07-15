#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tpcf.py
Tools to compute the two-point correlation functions accurately and
efficiently using analytic method.

Attributes
----------
    analytic_rr(rs, shape='rect', bounds=None)
    analytic_dr(D, rs, shape='rect', bounds=None)
    calculate_dd(D, r)
    analytic_tpcf(D, rs, shape='rect', bounds=None, est='nat')

Example
-------

Reference:
-------
He2021

Author
------
Chong-Chong He (che1234@umd.edu)

"""

import sys
import numpy as np
from numpy.linalg import norm
from math import pi, sqrt, asin, acos, atan
from scipy.spatial import cKDTree
from time import time
try:
    from numba import jit
except ModuleNotFoundError:
    pass
    sys.exit("""To use numba to speed up the code, please install numba (pip/conda
install numba). It is tested and shown that numba speed up dr_cuboid()
by 100 times. To preceed without using numba, comment out this line at the
beginning of tpcf.py and all the lines biginning with '@jit' """)

#------------------- Auxiliary functions -------------------------

def rr_rectangle(r, a, b):
    """Antiderivative of RR_rectangle. Equation (63) of He2021

    The RR from r1 to r2 for a rectangular area with sides a and b is
    rr_rectangle(r2, a, b) - rr_rectangle(r1, a, b)
    """
    return (2 * a * b * np.pi * r**2/2 - 4 * (a + b) * r**3/3 + 2 * r**4/4) / (a**2 * b**2)

def rr_cuboid(r, a, b, c):
    """Antiderivative of RR_cuboid. Equation (64) of He2021

    The RR from r1 to r2 for a cuboidal area with sides a and b is
    rr_cuboid(r2, a, b, c) - rr_rectangle(r1, a, b, c)
    """
    top = 4 * pi * a * b * c * r**3/3
    top += -2. * pi * (a*b + a*c + b*c) * r**4/4
    top += 8/3 * (a + b + c) * r**5/5
    top += -1. * r**6/6
    return top / (a*b*c)**2

def rr_unit_circle(r):
    """Antiderivative of RR_circle. Equation (65) of He2021

    The RR from r1 to r2 for a unit circular area is
    rr_unit_circle(r2) - rr_unit_circle(r1)
    """
    return r**2 - 4 / (3 * pi) * r**3 + 1 / (30 * pi) * r**5 + 1 / (7 * 160 * pi) * r**7

def rr_unit_sphere(r):
    """Antiderivative of RR_sphere. Equation (66) of He2021

    The RR from r1 to r2 for a unit spherical area is
    rr_unit_sphere(r2) - rr_unit_sphere(r1)
    """
    return r**3 - 9 / 16 * r**4 + 1 / 32 * r**6

@jit
def dr_rectangle(data, rs, a, b):
    """Compute DR_rectangle(r) in O(n) time. Figure A1 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a rectangular area with sides a and b, returns the DR in a list of
    length scales, rs.
    """

    def int_rec_edge(r, gap):
        """ (Happen to be) Normalized such that f = 0 when r = gap """
        return -1. * gap * sqrt(r**2 - gap**2) + r**2 * acos(gap/r)

    def int_rec_corner(r, xgap, ygap):
        """ Normalized such that f = 0 when r = sqrt(xgap^2 + ygap^2) """
        return 1. / 4 * (pi * r**2 - 2 * xgap * sqrt(r**2 - xgap**2) \
                         - 2 * ygap * sqrt(r**2 - ygap**2) \
                         - 2 * r**2 * (asin(xgap/r) + asin(ygap/r))) \
                         + xgap * ygap

    N = data.shape[0]
    DRhat = []
    rsteps = np.diff(rs)
    for i in range(len(rs) - 1):
        drpair = 0.0
        rthis = rs[i]
        rnext = rs[i + 1]
        r = rnext
        for j in range(N):
            drpair += pi * (rnext**2 - rthis**2)
            x, y = data[j, :]
            if x < r:
                xgap = x
            elif x > a - r:
                xgap = a - x
            else:
                xgap = -1.
            if y < r:
                ygap = y
            elif y > b - r:
                ygap = b - y
            else:
                ygap = -1.

            for igap in [xgap, ygap]:
                if igap > 0:
                    # drpair -= 2 * acos(xgap / r) * r
                    if igap >= rthis:
                        F1 = 0.
                    else:
                        F1 = int_rec_edge(rthis, igap)
                    F2 = int_rec_edge(rnext, igap)
                    drpair -= F2 - F1

            if xgap > 0 and ygap > 0 and xgap**2 + ygap**2 < r**2:
                # drpair += (pi/2 - asin(xgap/r) - asin(ygap/r)) * r
                if xgap**2 + ygap**2 >= rthis**2:
                    F1 = 0.
                else:
                    F1 = int_rec_corner(rthis, xgap, ygap)
                F2 = int_rec_corner(rnext, xgap, ygap)
                drpair += F2 - F1

        DRhat.append(drpair / (N * a * b))
    return DRhat

@jit
def dr_cuboid(data, rs, a, b, c):
    """Compute DR_cuboid(r) in O(n) time. Figure A2 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a cuboidal area with sides a, b, and c, returns the DR in a list of
    length scales, rs.
    """

    def int_cube_face(r, gap):
        """NOT normalized such that f = 0 when r = gap. Always use the
        difference between two r's."""
        return 2 * pi * (r**3/3 - gap * r * r / 2)

    def int_cube_edge(r, x, y):
        """ Normalized such that f = 0 when r = sqrt(x^2 + y^2)

        x, y should be better named as xgap, ygap, but I don't bother to do that.
        """

        # if abs((r*r - x*x - y*y)/(r*r)) < 1e-10:
        #     # limit of r^2 -> x^2 + y^2 and h = sqrt(r^2 - x^2 - y^2) -> 0+
        #     return pi / 3 * (x**3 + y**3)
        assert (r*r - x*x - y*y)/(r*r) > 0, \
            ("If you see this error message, it means you have discovered a"
             " flaw of this code. Please report it to the author "
             "(ChongChong He, che1234@umd.edu)")
        h = sqrt(r*r - x*x - y*y)
        part1 = 1/6 * pi * r**2 * (r-3 * (x+y)) \
            + 2 * x * y * h + y**3 * (pi/2 - atan(x/h)) \
            + x**3 * (pi/2 - atan(y/h)) \
            + r**2 * (y * atan(x/h) + x * atan(y/h))
        part2 = x**3 * (atan((r*x + x*x + y*y)/(h*y)) - atan((r*x - x*x - y*y)/(h*y))) \
            + r**3*atan((r**4 - r*r*x*x - r*r*y*y - x*x*y*y)/(2*x*y*r*h)) \
            + y**3 * (atan((r*y + x*x + y*y)/(h*x)) - atan((r*y - x*x - y*y)/(h*x))) \
            - 4*y*x*h
        part2 *= 1./3
        part_constant = -1. * pi / 3 * (x**3 + y**3)
        return part1 + part2 + part_constant

    N = data.shape[0]
    DRhat = []
    rsteps = np.diff(rs)
    for i in range(len(rs) - 1):
        drpair = 0.0
        rthis = rs[i]
        rnext = rs[i + 1]
        r = rnext
        for j in range(N):
            # drpair += 4 * pi * r * r
            drpair += 4. / 3 * pi * (rnext**3 - rthis**3)

            x, y, z = data[j, :]
            if x < r:
                xgap = x
            elif x > a - r:
                xgap = a - x
            else:
                xgap = -1.
            if y < r:
                ygap = y
            elif y > b - r:
                ygap = b - y
            else:
                ygap = -1.
            if z < r:
                zgap = z
            elif z > c - r:
                zgap = c - z
            else:
                zgap = -1.

            for igap in [xgap, ygap, zgap]:
                if igap > 0:
                    # drpair -= 2 * pi * (1 - igap/r) * r * r
                    r1 = igap if igap > rthis else rthis
                    drpair -= int_cube_face(rnext, igap) - int_cube_face(r1, igap)

            for gapi, gapj in [(xgap, ygap), (xgap, zgap), (ygap, zgap)]:
                if gapi > 0 and gapj > 0 and gapi**2 + gapj**2 < r**2:
                    # t = sqrt(1 - (gapi/r)**2 - (gapj/r)**2)
                    # omega = (0.5 - gapi/r - gapj/r) * pi + 2 * gapi/r * atan(gapj/r/t) + 2 * gapj/r * atan(gapi/r/t) + atan((t**2 - (gapi/r)**2 * (gapj/r)**2)/(2 * gapi/r * gapj/r * t))
                    # drpair += omega * r * r
                    if gapi**2 + gapj**2 >= rthis**2:
                        # r1 = sqrt(gapi**2 + gapj**2)
                        F1 = 0.0
                    else:
                        F1 = int_cube_edge(rthis, gapi, gapj)
                    F2 = int_cube_edge(rnext, gapi, gapj)
                    drpair += F2 - F1

        DRhat.append(drpair / (N * a * b * c))
    return DRhat

@jit
def dr_unit_circle(data, rs):
    """Compute DR_circle(r) in O(n) time. Figure A3 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a unit circle, returns the DR in a list of length scales, rs.
    """

    # make sure the circle has unitary radius
    assert norm(data, axis=1).max() <= 1.0
    assert rs.max() <= 1.0

    def int_unit_circle_edge(r, x):
        eta = sqrt((1 + x - r) * (1 + r - x) * (x + r - 1) * (x + r + 1))
        integral = pi*r**2 + eta/2. - r**2*acos((-1 + r**2 + x**2)/(2.*x*r)) \
            + atan((1 - r**2 + x**2)/eta)
        # constant = -(pi*(1.5 + (-2 + x)*x)) + (-1 + x)**2*acos(-(x/pi))
        constant = - pi / 2
        return integral + constant

    N = data.shape[0]
    DRhat = []
    rsteps = np.diff(rs)
    for i in range(len(rs) - 1):
        drpair = 0.0
        rthis = rs[i]
        rnext = rs[i + 1]
        r = rnext
        for j in range(N):
            drpair += pi * (rnext**2 - rthis**2)
            x = norm(data[j, :])
            # if x + r > 1:
            #     gap = 1. - x
            # else:
            #     gap = -1.
            gap = 1. - x
            # if gap > 0 and gap < r:
            if gap < r:
                # exclude edge effects
                if gap >= rthis:
                    F1 = 0.
                else:
                    F1 = int_unit_circle_edge(rthis, x)
                F2 = int_unit_circle_edge(rnext, x)
                drpair -= F2 - F1
        DRhat.append(drpair / (N * pi))
    return DRhat

@jit
def dr_unit_sphere(data, rs):
    """Compute DR_sphere(r) in O(n) time. Figure A4 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a unit sphere, returns the DR in a list of length scales, rs.
    """

    # make sure the circle has unitary radius
    assert norm(data, axis=1).max() <= 1.0
    assert rs.max() <= 1.0

    def int_unit_sphere_edge(r, x):
        return (pi*r**2*(-6 + 3*r**2 + 8*r*x + 6*x**2))/(12.*x)

    N = data.shape[0]
    DRhat = []
    rsteps = np.diff(rs)
    for i in range(len(rs) - 1):
        drpair = 0.0
        rthis = rs[i]
        rnext = rs[i + 1]
        r = rnext
        for j in range(N):
            drpair += 4. / 3 * pi * (rnext**3 - rthis**3)
            x = norm(data[j, :])
            gap = 1. - x
            if gap < r:
                # exclude edge effects
                if gap >= rthis:
                    F1 = int_unit_sphere_edge(gap, x)
                else:
                    F1 = int_unit_sphere_edge(rthis, x)
                F2 = int_unit_sphere_edge(rnext, x)
                drpair -= F2 - F1
        DRhat.append(drpair / (N * 4. / 3 * pi))
    return DRhat

#------------------- Main functions -------------------------

def analytic_rr(rs, shape='rect', bounds=None):
    """Returns the normalized RR in a list of length scales, rs.

    Args:
        rs (list): list of length scales at which RR is calculated
        shape (str): geometrical shape of the field,
            One of 'rect', 'cuboid', 'circle', 'sphere'.
        bounds (list): dimensions of the survey area in the format of
            [length, height] of a rectangle or [length, height, depth] of a
            cuboid. For circles and spheres, bounds is ignored.
    """

    if shape in ['rect', 'circle']:
        (a, b) = (1, 1) if bounds is None else bounds
    elif shape in ['cuboid', 'sphere']:
        (a, b, c) = (1, 1, 1) if bounds is None else bounds
    else:
        print("MyError: shape must be one of: rect, sphere, cuboid, sphere")
        return None
    if shape == 'rect':
        RRhat = rr_rectangle(rs[1:], a, b) - rr_rectangle(rs[:-1], a, b)
    elif shape == 'cuboid':
        RRhat = rr_cuboid(rs[1:], a, b, c) - rr_cuboid(rs[:-1], a, b, c)
    elif shape == 'circle':
        assert norm(D, axis=1).max() <= 1.0, \
            "All points should be inside a unit circle"
        RRhat = rr_unit_circle(rs[1:]) - rr_unit_circle(rs[:-1])
    elif shape == 'sphere':
        assert norm(D, axis=1).max() <= 1.0, \
            "All points should be inside a unit sphere"
        RRhat = rr_unit_sphere(rs[1:]) - rr_unit_sphere(rs[:-1])
    return RRhat

def analytic_dr(D, rs, shape='rect', bounds=None):
    """Returns the normalized DR in a list of length scales, rs.

    Args:
        D (2D array): N by 2 or 3 arrays of the coordinates of the galaxy particles.
        rs (list): a list of length scales at which RR is calculated
        shape (str): geometrical shape of the field,
            One of 'rect', 'cuboid', 'circle', 'sphere'.
        bounds (list): dimensions of the survey area in the format of
            [length, height] of a rectangle or [length, height, depth] of a
            cuboid. For circles and spheres, bounds is ignored.
    """

    if shape in ['rect', 'sphere']:
        (a, b) = (1, 1) if bounds is None else bounds
    elif shape in ['cuboid', 'sphere']:
        (a, b, c) = (1, 1, 1) if bounds is None else bounds
    else:
        print("MyError: shape must be one of: rect, sphere, cuboid, sphere")
        return None
    if shape == 'rect':
        DRhat = np.array(dr_rectangle(D, rs, a, b))
    elif shape == 'circle':
        DRhat = np.array(dr_unit_circle(D, rs))
    elif shape == 'cuboid':
        DRhat = np.array(dr_cuboid(D, rs, a, b, c))
    elif shape == 'sphere':
        DRhat = np.array(dr_unit_sphere(D, rs))
    return DRhat

def calculate_dd(D, r):
    """Brute-force calculation of normalized DD using scipy.cKDTree
    """

    D_tree = cKDTree(D)
    # count_neighbors includes self-self pair counts
    DD = D_tree.count_neighbors(D_tree, r, cumulative=False)[1:]
    return DD / D.shape[0]**2

def analytic_tpcf(D, rs, shape='rect', bounds=None, est='nat'):
    """Compute TPCF analytically based on the method presented in He2021.

    Args:
        D (2D array): array of galaxy particle coordinates. This must be
            a N by 2 or 3 array
        rs (array): array of scales at which the TPCF is calculated
        shape (str): one of ['rect', 'cuboid', 'circle', 'sphere']
        bound (list): the dimensions of the survay area. It is the length of
            the two sides of a rectangle, or the lengths of the three sides
            of a cuboid. When bound is None, unitary sides are assumed.
        est (str): the estimator to use. One of ['nat', 'LS']
    """

    t1 = time()
    RRhat = analytic_rr(rs, shape, bounds)
    dt = time() - t1
    print(f"Time spent on RR: {dt} sec")

    t1 = time()
    DDhat = calculate_dd(D, rs)
    dt = time() - t1
    print(f"Time spent on DD: {dt} sec")

    if est == 'nat':
        return DDhat / RRhat - 1
    elif est == 'LS':
        # compile analytic_dr with numba
        analytic_dr(D[:100, :], rs, shape, bounds)
        t1 = time()
        DRhat = analytic_dr(D, rs, shape, bounds)
        dt = time() - t1
        print(f"Time spent on DR: {dt} sec")
        return (DDhat - 2 * DRhat + RRhat) / RRhat

def MC_tpcf_mean_sigma(D, rs, bounds=None, n_catalogue=20, random_size=None, est='natural',
                       shape='rect', seed=None):
    """Compute two-point correlation functions using MC method and return
    the mean and sigma of the estimated xi(r) from n_catalogue random
    catalogues. This function is not perticular insteresting for this program.
    It is added here just for completeness and for testing and benchmarking
    our analytic method.

    Arg:
        D (2D array): array of particles showing their positions. This must be
            a n by 2 or 3 array
        rs (array): array of scales at which the TPCF is calculated
        bound (list): the dimensions of the survay area. It is the length of
            the two sides of a rectangle, or the lengths of the three sides
            of a cuboid. When bound is None, unitary sides are assumed.
        shape (str): one of ['rect', 'cuboid', 'circle', 'sphere']
        n_catalogue (int): the number of random catalogues. They are used to
            estimate the mean and std of the TPCF.

    Return:
        Mean and sigma of xi(r)
    """

    assert shape in ['rect', 'cuboid', 'circle', 'sphere']
    assert est in ['nat', 'natural', 'LS']
    n = D.shape[0]
    D_tree = cKDTree(D)
    DD = D_tree.count_neighbors(D_tree, rs, cumulative=False)[1:]  # including itself
    DDhat = DD / n**2
    if random_size is None:
        random_size = n
    if seed is not None:
        np.random.seed(seed)
    RRs = np.zeros([n_catalogue, len(rs) - 1])
    DRs = np.zeros([n_catalogue, len(rs) - 1])
    for i in range(n_catalogue):
        print(f"{i}/{n_catalogue}")
        if shape in ['rect', 'cuboid']:
            R = np.random.random([random_size, D.shape[1]])
            for j in range(len(bounds)):
                R[:, j] *= bounds[j]
        else:
            randomr = np.random.random(random_size)
            randomphi = np.random.random(random_size) * 2 * pi
            if D.shape[1] == 2:
                R = np.vstack((np.sqrt(randomr) * np.cos(randomphi),
                               np.sqrt(randomr) * np.sin(randomphi))).T
            elif D.shape[1] == 3:
                random_costheta = np.random.random(random_size) * 2 - 1  # from -1 to 1
                random_sintheta = np.sqrt(1 - random_costheta**2)
                R = np.vstack((
                    randomr**(1/3) * random_sintheta * np.cos(randomphi),
                    randomr**(1/3) * random_sintheta * np.sin(randomphi),
                    randomr**(1/3) * random_costheta)).T
        R_tree = cKDTree(R)
        RRs[i, :] = R_tree.count_neighbors(R_tree, rs, cumulative=False)[1:]
        DRs[i, :] = D_tree.count_neighbors(R_tree, rs, cumulative=False)[1:]
    RR_mean = np.mean(RRs, axis=0)
    RRhat = RR_mean / random_size**2
    RR_std = np.std(RRs, axis=0) / random_size**2
    DR_mean = np.mean(DRs, axis=0)
    DRhat = DR_mean / (n * random_size)
    DR_std = np.std(DRs, axis=0) / (n * random_size)
    if est in ['natural', 'nat']:
        xi = DDhat / RRhat - 1
        xi_std = RR_std / RRhat * (DDhat / RRhat)
        return xi, xi_std
    else:
        xi = (DDhat - 2 * DRhat + RRhat) / RRhat
        d_A2 = RR_std**2 + 4 * DR_std**2
        d_B2 = RR_std**2
        A = DDhat - DRhat + RRhat
        B = RRhat
        d_xi2 = (d_A2 / A**2 + d_B2 / B**2) * xi**2
        xi_std = np.sqrt(d_xi2)
        return xi, xi_std

