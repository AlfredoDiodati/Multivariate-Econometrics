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

def aug_dickey_fuller(x: np.ndarray, k: int) -> dict:
    """Augmented Dickey-Fuller Test"""
    
    dz = np.diff(z)
    if k == 0:
        # This part includes the simple Dickey-Fuller Test
        y = dz
        X = z[:-1][:, None]
        n_obs = len(y)

    else:
        # This is the Augmented part
        y = dz[k:]
        n_obs = len(y)
        
        # Y vector starting from t = k + 1
        x1_level = z[k:-1]
        
        X-lags = np.zeros((n_obs, k))
        for i in range(k):
            lag = i + 1
            X_lags[:, i] = dz[k-lag;-lag]

        X = np.column_stack([x1_level, X_lags])

    coeffs, ssr_list, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ coeffs
    ssr = np.sum(residuals**2)

    delta_hat = coeffs[0]

    k_params = X.shape[1]

    s2 = ssr / (n_obs - k_params)

    t_stat = delta_hat / se_delta

    return {
        'delta_hat': delta_hat,
        't_stat': t_stat,
        'residuals': residuals,
        'ssr': ssr,
        'n_obs': n_obs,
        'k_params;: k_params
    }





    
