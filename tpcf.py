#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tpcf.py
Some functions to compute the two-point correlation functions accurately and 
efficiently using an analytic method.

Attributes
----------
    rr_rectangle(rbins, a, b)
    rr_cuboid(rbins, a, b, c)
    rr_cube(rbins, a)
    rr_unit_circle(rbins)
    rr_unit_sphere(rbins)
    dr_rectangle(data, rs, a, b)
    dr_cuboid(data, rs, a, b, c)
    dr_unit_circle(data, rs)
    dr_unit_sphere(data, rs)
    dr_cube(data, rs, a)
    tpcf_ana(D, r, shape='rec', bound=None, est='nat')

Author
------
Chong-Chong He (che1234@umd.edu)
Date: 2021-08-01

If you use this code in your research, please consider citing the following paper:

https://ui.adsabs.harvard.edu/abs/2021arXiv210706918H/abstract

or

C.-C. He, 2021, "A Fast and Precise Analytic Method of Calculating Galaxy Two-point Correlation Functions", arXiv e-prints, arXiv:2107.06918.

Get help
--------

https://github.com/chongchonghe/analytic-2pcf

"""

import sys
import numpy as np
from numpy.linalg import norm
from math import pi, sqrt, asin, acos, atan
from scipy.spatial import cKDTree
from time import time
try:
    from numba import njit
except ModuleNotFoundError:
    numba_msg = """By default, this code use numba to speed up the dr_*
functions. Please install numba (pip install numba). Numba will speed
up dr_cuboid() by ~100 times. 

Press Enter to preceed without installing numba. Press C-c to quit.
"""
    print(numba_msg)
    input()
    print("Warning: running DR with pure Python. Gonna be slow!")
    def njit(func): return func  # this overwrites the Numba njit decorator.

#------------------- Auxiliary functions -------------------------

def Frr_rectangle(r, a, b):
    """Antiderivative of RR_rectangle.

    The RR from r1 to r2 for a rectangular area with sides a and b is
    rr_rectangle(r2, a, b) - rr_rectangle(r1, a, b)
    """
    return (2 * a * b * np.pi * r**2/2 - 4 * (a + b) * r**3/3 + 2 * r**4/4) / (a**2 * b**2)

def Frr_cuboid(r, a, b, c):
    """Antiderivative of RR_cuboid.

    The RR from r1 to r2 for a cuboidal area with sides a and b is
    rr_cuboid(r2, a, b, c) - rr_rectangle(r1, a, b, c)
    """
    top = 4 * pi * a * b * c * r**3/3
    top += -2. * pi * (a*b + a*c + b*c) * r**4/4
    top += 8/3 * (a + b + c) * r**5/5
    top += -1. * r**6/6
    return top / (a*b*c)**2

def Frr_unit_circle(r):
    """Antiderivative of RR_circle. 

    The RR from r1 to r2 for a unit circular area is
    rr_unit_circle(r2) - rr_unit_circle(r1)
    """
    return 2 * r**2/2 - 4 / pi * r**3/3 + 1 / (6 * pi) * r**5/5 + 1 / (160 * pi) * r**7/7

def Frr_unit_sphere(r):
    """Antiderivative of RR_sphere.

    The RR from r1 to r2 for a unit spherical area is
    rr_unit_sphere(r2) - rr_unit_sphere(r1)
    """
    return 3 * r**3/3 - 9 / 4 * r**4/4 + 3 / 16 * r**6/6


#------------------- Main functions -------------------------

def rr_rectangle(rbins, a, b):
    """ RR_rect(r; a, b) """
    return Frr_rectangle(rbins[1:], a, b) - Frr_rectangle(rbins[:-1], a, b)

def rr_cuboid(rbins, a, b, c):
    """ RR_cuboid(r; a, b, c) """
    return Frr_cuboid(rbins[1:], a, b, c) - Frr_cuboid(rbins[:-1], a, b, c)

def rr_cube(rbins, a):
    """ RR_cube(r; a) """
    return rr_cuboid(rbins, a, a, a)

def rr_unit_circle(rbins):
    """ RR_circle(r) """
    assert np.all(rbins <= 1.0) and np.all(rbins >= 0)
    return Frr_unit_circle(rbins[1:]) - Frr_unit_circle(rbins[:-1])

def rr_unit_sphere(rbins):
    """ RR_sphere(r) """
    assert np.all(rbins <= 1.0) and np.all(rbins >= 0)
    return Frr_unit_sphere(rbins[1:]) - Frr_unit_sphere(rbins[:-1])

@njit
def dr_rectangle(data, rs, a, b):
    """Compute DR_rectangle(r) in O(n) time. Figure A.1 of He2021.

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
            xgapl = x if x < r else -1.
            xgapr = a - x if x > a - r else -1.
            ygapl = y if y < r else -1.
            ygapr = b - y if y > b - r else -1.

            for igap in [xgapl, xgapr, ygapl, ygapr]:
                if igap > 0:
                    # drpair -= 2 * acos(xgap / r) * r
                    if igap >= rthis:
                        F1 = 0.
                    else:
                        F1 = int_rec_edge(rthis, igap)
                    F2 = int_rec_edge(rnext, igap)
                    drpair -= F2 - F1

            for (xgap, ygap) in [(xgapl, ygapl), (xgapl, ygapr), (xgapr, ygapl), (xgapr, ygapr)]:
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

@njit
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
            xgapl = x if x < r else -1.
            xgapr = a - x if x > a - r else -1.
            ygapl = y if y < r else -1.
            ygapr = b - y if y > b - r else -1.
            zgapl = z if z < r else -1.
            zgapr = c - z if z > c - r else -1.

            for igap in [xgapl, xgapr, ygapl, ygapr, zgapl, zgapr]:
                if igap > 0:
                    # drpair -= 2 * pi * (1 - igap/r) * r * r
                    r1 = igap if igap > rthis else rthis
                    drpair -= int_cube_face(rnext, igap) - int_cube_face(r1, igap)

            xgaps = (xgapl, xgapr)
            ygaps = (ygapl, ygapr)
            zgaps = (zgapl, zgapr)
            for (igaps, jgaps) in [(xgaps, ygaps), (xgaps, zgaps), (ygaps, zgaps)]:
                for gapi in igaps:
                    for gapj in jgaps:
                        if gapi > 0 and gapj > 0 and gapi**2 + gapj**2 < r**2:
                            if gapi**2 + gapj**2 >= rthis**2:
                                F1 = 0.0
                            else:
                                F1 = int_cube_edge(rthis, gapi, gapj)
                            F2 = int_cube_edge(rnext, gapi, gapj)
                            drpair += F2 - F1

        DRhat.append(drpair / (N * a * b * c))
    return DRhat

@njit
def dr_unit_circle(data, rs):
    """Compute DR_circle(r) in O(n) time. Figure A3 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a unit circle, returns the DR in a list of length scales, rs.
    """

    # make sure the circle has unitary radius
    # assert norm(data, axis=1).max() <= 1.0
    # assert rs.max() <= 1.0

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

@njit
def dr_unit_sphere(data, rs):
    """Compute DR_sphere(r) in O(n) time. Figure A4 of He2021.

    For a given dataset with the positions of a population of galaxies inside
    a unit sphere, returns the DR in a list of length scales, rs.
    """

    # make sure the circle has unitary radius
    # assert norm(data, axis=1).max() <= 1.0
    # assert rs.max() <= 1.0

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

def dr_cube(data, rs, a):
    return dr_cuboid(data, rs, a, a, )

def dr_general(D, r, shape, bound):
    if shape == 'rect':
        return dr_rectangle(D, r, *bound)
    elif shape == 'cuboid':
        return dr_cuboid(D, r, *bound)
    elif shape == 'circle':
        return dr_unit_circle(D, r)
    elif shape == 'sphere':
        return dr_unit_sphere(D, r)
    return

def calculate_dd(D, r):
    """(Brute-force) calculation of normalized DD using scipy.cKDTree
    """

    D_tree = cKDTree(D)
    # count_neighbors includes self-self pair counts
    DD = D_tree.count_neighbors(D_tree, r, cumulative=False)[1:]
    return DD / D.shape[0]**2

def tpcf_ana(D, r, shape='rec', bound=None, est='nat'):
    """Compute RR and DR analytically based on the method presented in He2021 
    and return the TPCF.
    
    RR and DR are computed analytically. DD is computed numerically using 
    scipy.spatial.cKDTree.

    Args:
        D (2D array): Array of galaxy particle coordinates. This must be
            an N by 2 or N by 3 array
        r (array): Array of scales at which the TPCF is calculated
        shape (str): One of ['rect', 'cuboid', 'circle', 'sphere']
        bound (list): The dimensions of the survay area. It is a tuple of 
            the lengths of the two sides of a rectangle or the three sides 
            of a cuboid. When bound is None, unitary sides are assumed. 
            When shape = 'circle' or 'sphere', bound is ignored.
        est (str): The estimator to use. One of ['nat', 'LS']
    
    Returns:
        xi (array): Array with length = len(r) - 1.
    """

    t1 = time()
    if shape == 'rect':
        RRhat = rr_rectangle(r, *bound)
    elif shape == 'cuboid':
        RRhat = rr_cuboid(r, *bound)
    elif shape == 'circle':
        RRhat = rr_unit_circle(r)
    elif shape == 'sphere':
        RRhat = rr_unit_sphere(r)
    dt = time() - t1
    print(f"Time spent on RR: {dt} sec")

    if est == 'LS':
        # compile analytic_dr with numba. This time is not reported in the benchmark
        dr_general(D[:min(100, D.shape[0]), :], r, shape, bound)
        t1 = time()
        DRhat = dr_general(D, r, shape, bound)
        dt = time() - t1
        print(f"Time spent on DR: {dt} sec")

    t1 = time()
    DDhat = calculate_dd(D, r)
    dt = time() - t1
    print(f"Time spent on DD: {dt} sec")

    if est == 'nat':
        return DDhat / RRhat - 1
    elif est == 'LS':
        return (DDhat - 2 * np.array(DRhat) + RRhat) / RRhat
    return
