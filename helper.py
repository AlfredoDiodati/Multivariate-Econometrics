import numpy as np
from scipy.stats import chi2

def acf(x) -> np.ndarray:
    """Autocorrelation function. Note: biased as in most packages."""
    n = len(x)
    x_center = x - np.nanmean(x)
    return (np.correlate(x_center, x_center, mode='full'))[n-1:] / (np.nanvar(x)*n)

def dickey_fuller(x: np.ndarray) -> tuple[float, float]:
    """Dickey Fuller test for unit root.
        Returns: 
            - regression coefficient (delta_hat)
            - t statistic (t_stat)
    """
    dx = np.diff(x)
    x_lag = x[:-1]
    delta_hat = np.linalg.lstsq(x_lag[:, None], dx, rcond=None)[0][0]
    u_hat = dx - delta_hat * x_lag
    s2_u = np.mean(u_hat**2)
    t_stat = delta_hat / np.sqrt(s2_u/np.sum(x_lag**2))
    return delta_hat, t_stat

def ljung_box(x:np.ndarray) -> tuple[float, float]:
    """Ljung–Box test statistic and p. values"""
    n = len(x)
    lags = np.arange(n)
    lj_bx = n*(n+2)*np.cumsum(acf(x)**2/np.arange(n, 0, -1))
    pv = 1 - chi2.cdf(lj_bx, lags)
    return lj_bx, pv