# Analytic Compution of the Two-point Correlation Functions 

If you use this code in your research, please consider citing the
following article:

(to be added later)

## Getting Started

### Dependencies

numpy, scipy, numba (optional)

### Executing program

The functions in `tpcf.py` can be used to compute the random-random
pair counts RR and the data-random pair counts DR analytically, using
`analytic_rr` and `analytic_dr`. A function `calculate_dd` is also
provided that computes the data-data pair counts DD using scipy.cKDTree.

For an example usage of the functions, check the `main` function in `main.py`. 

To try out this code, run `python main.py`, which will create a figure
of the TPCF of a mock galaxy. The time elapsed on DD, RR, and DR is
reported.

## Authors

Chong-Chong He  
che1234 at umd.edu  
<https://www.astro.umd.edu/~chongchong/>
