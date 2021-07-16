# Analytic Compution of the Two-point Correlation Functions 

If you use this code in your research, please consider citing the following article:

https://ui.adsabs.harvard.edu/abs/2021arXiv210706918H/abstract

## Getting Started

### Dependencies

numpy, scipy, numba (optional)

### Executing program

The functions in `tpcf.py` can be used to compute the random-random pair counts RR and the data-random pair counts DR analytically, using `analytic_rr` and `analytic_dr`. When accelerated by numba, they can deal with a million particles in under 5 seconds. A function `calculate_dd` is also provided that computes the data-data pair counts DD using scipy.cKDTree. Putting all together in `analytic_tpcf`, this function returns the TPCF for a given dataset and survey geometry with natural or LS estimator, achieving a speed 4 to 5 orders of magnitude faster than the brute-force Monte Carlo method.

For an example usage, check the `plot_test_tpcf` function in `test.py`. 

To try out this code, run `python test.py`, which will create a figure of the TPCF of a mock galaxy. The time elapsed on DD, RR, and DR will be reported.

## Authors

Chong-Chong He  
che1234 at umd.edu  
<https://www.astro.umd.edu/~chongchong/>
