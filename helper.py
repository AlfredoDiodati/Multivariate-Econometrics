import numpy as np
from scipy.stats import chi2

def acf(x):
    "Standard biased acf function"
    x = np.asarray(x)
    x_center = x - np.mean(x)
    n = len(x)
    acov = np.correlate(x_center, x_center, mode='full')[n-1:] / n
    return acov / acov[0]

def acf_sample(x):
    """Correct sample correction with unbiased denominator n-k"""
    x = np.asarray(x)
    x_center = x - np.mean(x)
    n = len(x)
    raw = np.correlate(x_center, x_center, mode='full')[n-1:]
    gamma = raw / (np.arange(n, 0, -1))
    return gamma / gamma[0]

def ar_coefficients(x:np.ndarray)->tuple[float, float]:
    """Return OLS estimated parameters of AR(1) model with intercept (intercept, slope)"""
    x_new = x[1:]
    x_lag = x[:-1]
    x_nmean = x_new.mean()
    x_lmean = x_lag.mean()
    x_lcentered = (x_lag - x_lmean)
    slope = np.sum(x_lcentered*(x_new - x_nmean))/np.sum(x_lcentered**2)
    intercept = x_nmean - slope*x_lmean
    return (intercept, slope)

def ar_standard_errors(x:np.ndarray)->tuple[float, float]:
    """Computes standard error of AR(1) with intercept with parameters"""
    intercept, slope = ar_coefficients(x)
    T = x.shape[0]
    x_new = x[1:]
    x_lag = x[:-1]
    x_lmean = x_lag.mean()
    x_lss = np.sum((x_lag - x_lmean)**2)
    residuals = x_new - intercept - x_lag * slope
    res_var = np.sum(residuals**2)/(T - 2.0)
    se_slope = np.sqrt(res_var/x_lss)
    se_intercept = np.sqrt(res_var*(1/(T-1.0)+x_lmean**2/x_lss))
    return (se_intercept, se_slope)

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
    s2_u = np.sum(u_hat**2) / (len(x_lag) - 1)
    t_stat = delta_hat / np.sqrt(s2_u/np.sum(x_lag**2))
    return delta_hat, t_stat

def ljung_box(x:np.ndarray) -> tuple[float, float]:
    """Ljung-Box test statistic and p. values"""
    n = len(x)
    lags = np.arange(n)
    lj_bx = n*(n+2)*np.cumsum(acf(x)**2/np.arange(n, 0, -1))
    pv = 1 - chi2.cdf(lj_bx, lags)
    return lj_bx, pv

def aug_dickey_fuller(z: np.ndarray, k: int) -> dict:
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
        
        X_lags = np.zeros((n_obs, k))

        for i in range(k):
            lag = i + 1
            X_lags[:, i] = dz[k-lag:-lag]

        X = np.column_stack([x1_level, X_lags])

    coeffs, ssr_list, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ coeffs
    ssr = np.sum(residuals**2)

    delta_hat = coeffs[0]

    k_params = X.shape[1]

    s2 = ssr / (n_obs - k_params)

    X_inv = np.linalg.inv(X.T @ X)

    var_delta = s2 * X_inv[0, 0]
    se_delta = np.sqrt(var_delta)

    t_stat = delta_hat / se_delta

    return {
        'delta_hat': delta_hat,
        't_stat': t_stat,
        'residuals': residuals,
        'ssr': ssr,
        'n_obs': n_obs,
        'k_params': k_params
    }