# Analytic Calculation of the Two-point Correlation Functions 

Some functions to compute the two-point correlation functions accurately and 
efficiently using an analytic method. For more details of this method, check 
https://ui.adsabs.harvard.edu/abs/2021arXiv210706918H/abstract

## Getting Started

### Dependencies

numpy, scipy, numba (optional)

### Executing program

The functions in `tpcf.py` can be used to compute the random-random pair counts RR and the data-random pair counts DR analytically in O(1) and O(N) time, respectively.  With numba installed, the code can deal with 10 million particles in under 1 minute on a single core. 

For an example usage, check the `plot_test_tpcf` function in `test.py`. 

To try out this code, run `python test.py`, which will create a figure of the TPCF of a mock galaxy. The time elapsed on DD, RR, and DR will be reported.

## Authors

Chong-Chong He  
che1234 at umd.edu  
<https://www.astro.umd.edu/~chongchong/>

If you use this code in your research, please consider citing the following article:

https://ui.adsabs.harvard.edu/abs/2021arXiv210706918H/abstract
