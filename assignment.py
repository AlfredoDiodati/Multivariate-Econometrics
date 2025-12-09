import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, f
from scipy.stats import norm
import statsmodels.api as sm
print("Warning! This code is supposed to be runned directly and from the same directory as the data\n")

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

rng: np.random.Generator = np.random.default_rng(seed = 121)

# Task 1.1
T = 500
error_variance = 1.0
errors = rng.normal(loc=0.0, scale=error_variance, size=T + 50)
random_walk = np.cumsum(errors)[50:]

ar_variance = 1.0
ar_coefficient = 0.7
ar_errors = rng.normal(loc=0.0, scale=ar_variance, size=550)[50:]
ar = np.convolve(ar_errors, ar_coefficient**np.arange(T), mode='full')[:T]

acf_rw = acf(random_walk)
acf_ar = acf(ar)
lags = np.arange(acf_rw.shape[0])


fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2 rows, 1 column

axes[0].plot(random_walk, label="Random Walk")
axes[0].plot(ar, label=rf"AR(1) $\rho = {ar_coefficient}$")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].set_title("Random Walk and AR(1) processes")
axes[0].legend()

axes[1].bar(lags, acf_rw, label="Random Walk")
axes[1].bar(lags, acf_ar, label=rf"AR(1) $\rho = {ar_coefficient}$")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("ACF")
axes[1].set_title(r"Autocorrelation of $y_t$ and $x_t$")
axes[1].legend()
plt.tight_layout()
plt.show()

    
# task 1.2

intercept, slope = ar_coefficients(random_walk)
se_intercept, se_slope = ar_standard_errors(random_walk)
t_intercept = intercept / se_intercept
t_slope = slope / se_slope
pv_intercept = 2 * (1 - norm.cdf(abs(t_intercept)))
pv_slope = 2 * (1 - norm.cdf(abs(t_slope)))


print(f"Estimated intercept {intercept.__round__(3)} with standard error {se_intercept.__round__(3)}")
print(f"Estimated slope {slope.__round__(3)} with standard error {se_slope.__round__(3)}")
print(f"Intercept and Slope t statistics: {t_intercept.__round__(3)} and {t_slope.__round__(3)} respectively")
print(f"...and corresponding p.values: {pv_intercept.__round__(3)} and {pv_slope.__round__(3)} respectively")
    
residuals = random_walk[1:] - intercept - slope * random_walk[:-1]
R2 = 1.0 - np.sum(residuals**2)/np.sum((random_walk[1:] - np.mean(random_walk[1:]))**2)
acf_residuals = acf(residuals)
    
print(fr"\n AR(1) with intercept's R^2: {R2.__round__(3)}")

plt.bar(np.arange(acf_residuals.shape[0]), acf_residuals)
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.title("ACF of residuals of Estimated AR(1) with intercept")
plt.show()
    
lb_stat, lb_pv = ljung_box(residuals)

lags = np.arange(1, 20)
lb_stat_filtered, lb_pv_filtered = lb_stat[lags], lb_pv[lags]

lb_table = pd.DataFrame({
    'Ljung-Box Statistics': lb_stat_filtered,
    'Lag': lags
})

pv_table = pd.DataFrame({
    'Ljung-Box p. value': lb_pv_filtered,
    'Lag': lags
})

print(f"Ljung-Box test statistics: \n{lb_table}")
print(f"\nLjung-Box test statistics: \n{pv_table}")


plt.plot(residuals)
plt.xlabel("Time")
plt.ylabel(r"Value")
plt.title("Residuals of estimated AR(1) with intercept")
plt.show()
    
# Task 1.3
_, df_stat = dickey_fuller(random_walk)
df_critical_v = -1.62
isrejected = df_stat < df_critical_v

print(f"\n Dickey Fuller statistics of Random Walk {df_stat.__round__(3)}")
if isrejected: print(f"Since {df_stat} < {df_critical_v} we reject the null at a 10% confidence interval")
else: print(f"Since {df_stat.__round__(3)} > {df_critical_v} we fail to reject the null at a 10% confidence interval")
    
first_differences = random_walk[1:]-random_walk[:-1]
df_coeff_diff, df_stat_diff = dickey_fuller(first_differences)

print(f"\n First differences of random walk \nDF coefficient {df_coeff_diff.__round__(3)} and statistics {df_stat_diff.__round__(3)}")
print(f"Therefore the null hpt. of a unit root is rejected, and we conclude the random walk is I(1) \n")
    
df_coeff_ar, df_stat_ar = dickey_fuller(ar)

print(f"Stationary AR(1) \n DF coefficient {df_coeff_ar.__round__(3)} and statistics {df_stat_ar.__round__(3)}")
print(f"Therefore the null hpt. of a unit root is rejected.")

# QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/
# QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/
# QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/ QUESTION 2/

rng: np.random.Generator = np.random.default_rng(seed = 123)
ismain: bool = __name__ == '__main__'

# Task 2.1

innovation_variance = 1.0
innovation = rng.normal(loc=0.0, scale=innovation_variance, size=550)
error_ar_coefficient = 0.5

correlated_errors = np.convolve(innovation, error_ar_coefficient**np.arange(550), mode='full')[:550]
correlated_random_walk = np.cumsum(correlated_errors)[50:]

first_difference = np.diff(correlated_random_walk)
len_series = len(first_difference)
lags = np.arange(len_series)
acf_crw = acf(first_difference)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2 rows, 1 column
axes[0].plot(correlated_random_walk)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].set_title("Random Walk with serially correlated innovations")

axes[1].bar(lags, acf_crw, label="Random Walk")
axes[1].bar(lags, acf_crw)
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("ACF")
axes[1].set_title(r"Autocorrelation of $\Delta z_t$")
plt.tight_layout()
plt.show()

cont_variance = innovation_variance/(1.0 - error_ar_coefficient**2) # see report for derivation

print(f"Contemporaneous variance: {cont_variance}")
print(f"Long run variance: {3 * cont_variance}") # see report for derivation

# Task 2.2
df_coeff, df_stat = dickey_fuller(correlated_random_walk)

print(f"""Dickey Fuller estimated coefficient {df_coeff.__round__(4)} and t-statistic {df_stat.__round__(2)} \n
Against the critical values 1% -2.58, 5% -1.95, 10% -1.62 \n,
Since {df_stat.__round__(2)} > -1.95 we fail to reject the null hypothesis.
""")

df_residuals = first_difference - df_coeff * correlated_random_walk[:-1]

plt.plot(df_residuals)
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.title(r"Dickey-Fuller Regression Residuals$")
plt.show()

lb_stat, lb_pv = ljung_box(df_residuals)

lags = np.arange(1, 20)
lb_stat_filtered, lb_pv_filtered = lb_stat[lags], lb_pv[lags]

lb_table = pd.DataFrame({
    ' Ljung-Box Statistic': lb_stat_filtered,
    'Lag': lags
})

pv_table = pd.DataFrame({
    'Ljung-Box p. value': lb_pv_filtered,
    'Lag': lags
})

print(f"Ljung-Box test statistics: \n{lb_table}")
print(f"\nLjung-Box test statistics: \n{pv_table}")
    
# Task 2.3

def calc_aic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0: return np.inf
    return 2 * k_params + n_obs * np.log(ssr / n_obs)

def calc_bic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0: return np.inf
    return k_params * np.log(n_obs) + n_obs * np.log(ssr / n_obs)

max_k=15
k_values = range(max_k + 1)
aic_scores = []
bic_scores = []

for k in k_values:
    results = aug_dickey_fuller(correlated_random_walk, k=k)
    if results:
        aic = calc_aic(results['n_obs'], results['ssr'], results['k_params'])
        bic = calc_bic(results['n_obs'], results['ssr'], results['k_params'])
    else:
        aic = np.inf
        bic = np.inf

    aic_scores.append(aic)
    bic_scores.append(bic)

best_k_aic = np.argmin(aic_scores)
best_k_bic = np.argmin(bic_scores)

print(f"Best k value (by AIC): {best_k_aic}")
print(f"Best k value (by BIC): {best_k_bic}")


best_k = best_k_aic if best_k_aic < best_k_bic else best_k_bic

final_adf = aug_dickey_fuller(correlated_random_walk, k=best_k)

adf_coeff = final_adf['delta_hat']
adf_tstat = final_adf['t_stat']

print(f"ADF coefficient: {adf_coeff}")
print(f"ADF t-statistic: {adf_tstat}")

print(f"""Dickey Fuller estimated coefficients {adf_coeff} and t-statistic {adf_tstat} \n
    Against the critical values 1% -2.58, 5% -1.95, 10% -1.62 \n,
    Since {adf_tstat} > -1.95 we fail to reject the null hypothesis.
    """)

adf_residuals = final_adf['residuals']
acf_adf_residuals = acf(adf_residuals)
plt.figure(figsize=(15,6))
plt.bar(range(1, len(acf_adf_residuals)), acf_adf_residuals[1:])
plt.xlabel("Lag")
plt.ylabel("Unbiased ACF")
plt.title(f"ACF of Residuals from ADF(k={best_k})")
plt.show()

lb_stat_adf, lb_pv_adf = ljung_box(adf_residuals)
failed_adf = np.sum(lb_pv_adf <= 0.05)

print(f"Ljung-Box Test: {failed_adf} out of {len(lb_pv_adf)} tests failed (p<=0.05).")
print(f"Since we failed {failed_adf} out of {len(lb_pv_adf)}, we have not successfully removed serial correlation.")

# Task 2.4
u_hat = df_residuals
T = len(u_hat)

# 2.4.2: Estimating the long run variance (Newey-West)
# bandwidth given by the exercise
q = int(np.floor(T**(1/3)))
autocovar = []

for j in range(q+1):
    u_t = u_hat[j:]
    u_t_minus_j = u_hat[:-j] if j > 0 else u_hat

    gamma_j = (1/T) * np.sum(u_t * u_t_minus_j)
    autocovar.append(gamma_j)

sigma_0_sq = autocovar[0]

sigma_u_sq = sigma_0_sq
for j in range(1, q+1):
    bartlett_weight = (1-j/(q+1))
    sigma_u_sq += 2* bartlett_weight * autocovar[j]

print(f"\nLong-run variance (sigma_u_sq): {sigma_u_sq}")
print(f"Simple variance (sigma_0_sq): {sigma_0_sq}")

if sigma_u_sq > sigma_0_sq:
    print("\nsigma_u^2 > sigma_0^2, confirming strong positive serial correlation")
    print("in the simple DF residuals (which we already knew from Ljung-Box).")
else:
    print("sigma_u^2 <= sigma_0^2, which is unexpected for this data.")

# 2.4.3
se_delta = df_coeff / df_stat
tstatpp = (df_stat * (sigma_0_sq**0.5 / sigma_u_sq**0.5)) - ((sigma_u_sq - sigma_0_sq) * T * se_delta)/ ( 2* (sigma_u_sq**0.5)*(sigma_0_sq**0.5))

print(f"\nPP corrected t statistic: {tstatpp}")

# QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/
# QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/
# QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/QUESTION 3/

# Load data
data = pd.read_csv('./VU_MVE_assignment_timeseries_data.csv')
data['Period'] = pd.PeriodIndex(data['Period'], freq='Q').to_timestamp()
data = data.set_index('Period')
data.index.name = None

# 3.1.1 Plot time series
for col, ylabel in [("GDP", "log GDP"), ("INFLATION", "%"), ("INTEREST_RATE", "%")]:
    plt.figure(figsize=(12,4))
    data[col].plot()
    plt.title(f"{col.replace('_', ' ')}")
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
# Plot ACF correlograms for all three series
maxlag = 20

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, series) in enumerate([("GDP", data["GDP"]), 
                                       ("INFLATION", data["INFLATION"]), 
                                       ("INTEREST_RATE", data["INTEREST_RATE"])]):
    # Compute ACF
    rho = acf(series.values)
    k = min(maxlag, len(rho) - 1)
    lags = np.arange(0, k + 1)
    
    # Plot correlogram
    axes[idx].bar(lags, rho[:k+1], alpha=0.7, color='steelblue')
    axes[idx].axhline(0, color='black', linewidth=0.8)
    
    # Add confidence bands (approximate 95% CI: ±1.96/√n)
    n = len(series)
    conf_level = 1.96 / np.sqrt(n)
    axes[idx].axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[idx].axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[idx].set_xlabel('Lag')
    axes[idx].set_ylabel('Autocorrelation')
    axes[idx].set_title(f'ACF: {name}')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 3.1.2-3.1.3 ADF tests with lag selection

def calc_aic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0:
        return np.inf
    return 2 * k_params + n_obs * np.log(ssr / n_obs)

def calc_bic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0:
        return np.inf
    return k_params * np.log(n_obs) + n_obs * np.log(ssr / n_obs)

def _adf_design(y: np.ndarray, k: int, det: str):
    """Build design matrix for ADF regression. det: 'nc'/'c'/'ct'"""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    y_lag = y[:-1]

    if k > 0:
        dY_lags = np.column_stack([dy[j:-k+j] for j in range(k)])
    else:
        dY_lags = None

    if k > 0:
        y_dep = dy[k:]
        y_lag_aligned = y_lag[k:]
    else:
        y_dep = dy
        y_lag_aligned = y_lag

    T_eff = len(y_dep)
    cols = []
    if det in ('c','ct'):
        cols.append(np.ones(T_eff))
    if det == 'ct':
        cols.append(np.arange(1, T_eff+1, dtype=float))
    
    cols.append(y_lag_aligned)
    if k > 0:
        cols.append(dY_lags)

    X = np.column_stack([c if c.ndim == 1 else c for c in cols]) if len(cols) else None
    return y_dep, X

def _ols_fit(y_dep: np.ndarray, X: np.ndarray):
    XTX = X.T @ X
    XTy = X.T @ y_dep
    beta = np.linalg.lstsq(XTX, XTy, rcond=None)[0]
    resid = y_dep - X @ beta
    ssr = float(resid.T @ resid)
    n = len(y_dep)
    k_params = X.shape[1]
    s2 = ssr / n
    cov_beta = s2 * np.linalg.pinv(XTX)
    se = np.sqrt(np.diag(cov_beta))
    return beta, se, resid, ssr, n, k_params

def adf_manual(y: np.ndarray, k: int, det: str):
    """Run ADF test with k lags and specified deterministic terms"""
    y_dep, X = _adf_design(y, k=k, det=det)
    beta, se, resid, ssr, n, k_params = _ols_fit(y_dep, X)

    idx_delta = {'nc': 0, 'c': 1, 'ct': 2}[det]
    delta_hat = beta[idx_delta]
    t_stat = delta_hat / se[idx_delta]
    
    return {
        "det": det, "k": k,
        "delta_hat": float(delta_hat),
        "t_stat": float(t_stat),
        "residuals": resid,
        "ssr": ssr, "n_obs": n, "k_params": k_params
    }

def adf_select_ic(y: np.ndarray, det: str, kmax: int = 12, ic: str = "AIC"):
    rows = []
    for k in range(kmax+1):
        try:
            res = adf_manual(y, k=k, det=det)
            if ic.upper() == "AIC":
                ic_val = calc_aic(res["n_obs"], res["ssr"], res["k_params"])
            else:
                ic_val = calc_bic(res["n_obs"], res["ssr"], res["k_params"])
            rows.append({**res, "ic": ic.upper(), "ic_val": ic_val})
        except Exception as e:
            rows.append({"det": det, "k": k, "ic": ic.upper(), "error": str(e)})
    
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["ic_val"])
    return best, rows

# Critical values (5% significance)
CRIT_5 = {"nc": -1.95, "c": -2.86, "ct": -3.41}

def adf_decision_5pct(t_stat: float, det: str) -> bool:
    return (t_stat is not None) and (t_stat < CRIT_5[det])

# Run tests
series_dict = {
    "GDP": data["GDP"].values.astype(float),
    "INFLATION": data["INFLATION"].values.astype(float),
    "INTEREST_RATE": data["INTEREST_RATE"].values.astype(float),
}

results_rows = []

for name, y in series_dict.items():
    for det in ["nc","c","ct"]:
        for ic in ["AIC","BIC"]:
            best, all_rows = adf_select_ic(y, det=det, kmax=12, ic=ic)
            if best is None:
                results_rows.append({
                    "series": name, "det": det, "ic": ic,
                    "k_selected": np.nan, "t_stat": np.nan, "delta_hat": np.nan,
                    "n_obs": np.nan, "ssr": np.nan, "decision_5%": "error"
                })
                continue
            decision = "reject H0" if adf_decision_5pct(best["t_stat"], det) else "fail to reject"
            results_rows.append({
                "series": name,
                "det": best["det"],
                "ic": best["ic"],
                "k_selected": best["k"],
                "t_stat": round(best["t_stat"], 3),
                "delta_hat": round(best["delta_hat"], 6),
                "n_obs": best["n_obs"],
                "ssr": round(best["ssr"], 6),
                "IC value": round(best["ic_val"], 3),
                "decision_5%": decision
            })

adf_table = pd.DataFrame(results_rows)
print(adf_table.sort_values(["series","det","ic"]).reset_index(drop=True))

# 3.1.4 Determine order of integration
def pick_best_for(series: str, det_pref=("ct","c","nc")):
    sub = adf_table[(adf_table["series"]==series) & (adf_table["ic"]=="AIC")]
    for det in det_pref:
        m = sub[sub["det"]==det]
        if len(m):
            return m.loc[m["IC value"].idxmin()]
    return None

order_rows = []
for s in ["GDP","INFLATION","INTEREST_RATE"]:
    lvl = pick_best_for(s)
    d = np.diff(series_dict[s])
    
    diff_rows = []
    for det in ["nc","c","ct"]:
        for ic in ["AIC","BIC"]:
            best_d, _ = adf_select_ic(d, det=det, kmax=12, ic=ic)
            if best_d:
                diff_rows.append(best_d)
    
    best_diff = min(diff_rows, key=lambda r: r.get("ic_val", np.inf))
    diff_decision = "reject H0" if adf_decision_5pct(best_diff["t_stat"], best_diff["det"]) else "fail to reject"
    
    if lvl is not None and lvl["decision_5%"] == "reject H0":
        order = "I(0)"
    elif diff_decision == "reject H0":
        order = "I(1)"
    else:
        order = "I(2) or higher"
    
    order_rows.append({
        "Series": s,
        "Level decision": lvl["decision_5%"] if lvl is not None else "N/A",
        "1st diff decision": diff_decision,
        "Order": order
    })

order_table = pd.DataFrame(order_rows)
print(order_table)

# 3.2 Spurious Regression (GDP on INFLATION, levels)
y = data['GDP'].values.astype(float)
x = data['INFLATION'].values.astype(float)

X = np.column_stack((np.ones(len(x)), x))
beta = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat = X @ beta
resid = y - y_hat

n = len(y)
k = X.shape[1]
ssr = float(np.sum(resid**2))
s2 = ssr / (n - k)
cov_beta = s2 * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(cov_beta))
t_beta = beta / se_beta
r2 = 1 - ssr / np.sum((y - y.mean())**2)

print("=== GDP ~ INFLATION (levels) — OLS results ===")
print(f"alpha (const)  = {beta[0]:.6f}  SE={se_beta[0]:.6f}  t={t_beta[0]:.2f}")
print(f"alpha1 (slope) = {beta[1]:.6f}  SE={se_beta[1]:.6f}  t={t_beta[1]:.2f}")
print(f"R^2            = {r2:.4f}")
print(f"n={n}, k={k}, SSR={ssr:.4f}")

# Scatter plot with fitted line
fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
ax.set_facecolor('#f8f9fa')
ax.scatter(x, y, alpha=0.7, s=60, c='#3498db', 
           edgecolors='white', linewidths=1.5, zorder=3, label='Observed')

xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
ax.plot(xs, beta[0] + beta[1]*xs, color='#e74c3c', linewidth=3, 
        alpha=0.9, zorder=2, label='Fitted')
ax.plot(xs, beta[0] + beta[1]*xs, color='#c0392b', linewidth=5, 
        alpha=0.2, zorder=1)

ax.set_xlabel("INFLATION (%)", fontsize=12, fontweight='bold')
ax.set_ylabel("GDP (log)", fontsize=12, fontweight='bold')
ax.set_title("GDP on INFLATION (levels)", fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Residual time plot
fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
ax.set_facecolor('#f8f9fa')

colors = ['#e74c3c' if r < 0 else '#3498db' for r in resid]
ax.scatter(data.index, resid, c=colors, alpha=0.6, s=30, edgecolors='white', 
           linewidths=0.5, zorder=3)
ax.plot(data.index, resid, color='#34495e', linewidth=1.5, alpha=0.5, zorder=2)

ax.axhline(0, color='black', linewidth=2, linestyle='-', alpha=0.8, zorder=1)
ax.fill_between(data.index, 0, resid, where=(resid > 0), 
                alpha=0.15, color='#3498db', interpolate=True)
ax.fill_between(data.index, 0, resid, where=(resid < 0), 
                alpha=0.15, color='#e74c3c', interpolate=True)

ax.set_title("Residuals from GDP ~ INFLATION", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Time", fontsize=12, fontweight='bold')
ax.set_ylabel("Residual", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Residual ACF
rho_full = acf(resid)
kbar = int(min(maxlag, len(rho_full)-1))
lags = np.arange(1, kbar+1)
heights = rho_full[1:kbar+1]

fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
ax.set_facecolor('#f8f9fa')

bars = ax.bar(lags, heights, alpha=0.85, color='#3498db', edgecolor='#2c3e50', 
              linewidth=1.2, zorder=3)

n_resid = len(resid)
conf_level = 1.96 / np.sqrt(n_resid)
for i, (lag, height) in enumerate(zip(lags, heights)):
    if abs(height) > conf_level:
        bars[i].set_edgecolor('#e74c3c')
        bars[i].set_linewidth(2.5)
        bars[i].set_alpha(1.0)

ax.axhline(0, color='black', linewidth=1.5, zorder=2)
ax.axhline(conf_level, color='#e74c3c', linestyle='--', linewidth=2, 
           alpha=0.8, label='95% CI', zorder=1)
ax.axhline(-conf_level, color='#e74c3c', linestyle='--', linewidth=2, 
           alpha=0.8, zorder=1)
ax.fill_between(lags, conf_level, 1, alpha=0.08, color='#e74c3c')
ax.fill_between(lags, -conf_level, -1, alpha=0.08, color='#e74c3c')

ax.set_title("Residual ACF: GDP ~ INFLATION", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Lag", fontsize=12, fontweight='bold')
ax.set_ylabel("Autocorrelation", fontsize=12, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, kbar + 1)
plt.tight_layout()
plt.show()

def ols_fit(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    n = len(y)
    k = X.shape[1]
    ssr = float(resid.T @ resid)
    s2 = ssr / (n - k)
    covb = s2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(covb))
    r2 = 1 - ssr / np.sum((y - np.mean(y))**2)
    return beta, y_hat, resid, se_beta, r2, ssr, s2, n, k

def dickey_fuller_simple(x: np.ndarray) -> tuple[float, float]:
    """Simple DF: delta x_t = δ·x_{t-1} + u_t"""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    x_lag = x[:-1]
    delta_hat = np.linalg.lstsq(x_lag[:, None], dx, rcond=None)[0][0]
    u_hat = dx - delta_hat * x_lag
    s2_u = np.mean(u_hat**2)
    t_stat = delta_hat / np.sqrt(s2_u / np.sum(x_lag**2))
    return float(delta_hat), float(t_stat)

def ljung_box_m(x, m=20):
    """Ljung-Box test for autocorrelation"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = acf(x)
    k = int(min(m, len(r)-1, n-1))
    lags = np.arange(1, k+1)
    Q = n*(n+2) * np.sum((r[1:k+1]**2) / (n - lags))
    p = 1 - chi2.cdf(Q, k)
    return Q, p, k

# Ljung-Box test
Q10, p10, k_used = ljung_box_m(resid, m=10)
print(f"Ljung-Box Q({k_used}) = {Q10:.2f},  p-value = {p10:.3f}")

# DF test on residuals
delta_hat_res, t_stat_res = dickey_fuller_simple(resid)
crit_no_const_5 = -1.95
decision = "REJECT H0 (stationary)" if t_stat_res < crit_no_const_5 else "FAIL TO REJECT (unit root)"
print("=== Simple DF on residuals ===")
print(f"delta_hat = {delta_hat_res:.6f},  t-stat = {t_stat_res:.3f},  5% crit ≈ {crit_no_const_5}")
print(f"Decision: {decision}")

# 3.3 Cointegration Analysis (Engle-Granger + ECM)

# Data
Y = data['GDP'].values.astype(float)
IR = data['INTEREST_RATE'].values.astype(float)

# Step 1: Cointegrating regression
X_c = np.column_stack((np.ones(len(IR)), IR))
beta_c, yhat_c, uhat, se_c, r2_c, ssr_c, s2_c, n_c, k_c = ols_fit(Y, X_c)

print("=== EG Step 1: Cointegrating regression (GDP on INTEREST_RATE, levels) ===")
print(f"beta0 = {beta_c[0]:.6f}  (SE={se_c[0]:.6f})")
print(f"beta1 = {beta_c[1]:.6f}  (SE={se_c[1]:.6f})")
print(f"R^2   = {r2_c:.4f},  n={n_c}, k={k_c},  SSR={ssr_c:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
ax.set_facecolor('#f8f9fa')
ax.scatter(IR, Y, alpha=0.7, s=60, c='#3498db', 
           edgecolors='white', linewidths=1.5, zorder=3, label='Observed')

xs = np.linspace(np.nanmin(IR), np.nanmax(IR), 200)
ax.plot(xs, beta_c[0] + beta_c[1]*xs, color='#e74c3c', linewidth=3, 
        alpha=0.9, zorder=2, label='Fitted')
ax.plot(xs, beta_c[0] + beta_c[1]*xs, color='#c0392b', linewidth=5, 
        alpha=0.2, zorder=1)

ax.set_xlabel("INTEREST_RATE (%)", fontsize=12, fontweight='bold')
ax.set_ylabel("GDP (log)", fontsize=12, fontweight='bold')
ax.set_title("Cointegrating Regression: GDP ~ INTEREST_RATE", fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Step 2: DF test on residuals
delta_u, tstat_u = dickey_fuller_simple(uhat)
CRIT_EG_5_const = -3.34
reject_cointegration_null = tstat_u < CRIT_EG_5_const

print("\n=== EG Step 2: DF test on residuals û_t ===")
print(f"DF t-stat = {tstat_u:.3f}  |  5% EG crit ≈ {CRIT_EG_5_const}")
print("Decision:", "REJECT — cointegration detected"
      if reject_cointegration_null else "FAIL TO REJECT — no cointegration")

# Error Correction Model
if reject_cointegration_null:
    dGDP = np.diff(Y)
    dIR  = np.diff(IR)
    ECM_lag1 = uhat[:-1]
    dGDP_lag1 = np.concatenate([[np.nan], dGDP[:-1]])
    
    df_ecm = pd.DataFrame({
        "dGDP": dGDP,
        "ECM_lag1": ECM_lag1,
        "dGDP_lag1": dGDP_lag1[1:],
        "dIR": dIR
    }).dropna()
    
    y_ecm = df_ecm["dGDP"].values
    X_ecm = np.column_stack((
        np.ones(len(df_ecm)),
        df_ecm["ECM_lag1"].values,
        df_ecm["dGDP_lag1"].values,
        df_ecm["dIR"].values
    ))

    beta_e, yhat_e, resid_e, se_e, r2_e, ssr_e, s2_e, n_e, k_e = ols_fit(y_ecm, X_ecm)

    print("\n=== ECM (ΔGDP on [const, ECM_{t-1}, ΔGDP_{t-1}, ΔIR]) ===")
    names = ["alpha0","alpha1(ECM_{t-1})","gamma1(ΔGDP_{t-1})","delta1(ΔIR)"]
    for nm, b, se in zip(names, beta_e, se_e):
        print(f"{nm:>22s} = {b:.6f}   (SE={se:.6f})")
    print(f"R^2 = {r2_e:.4f},  n={n_e}, k={k_e},  SSR={ssr_e:.4f}")

    if beta_e[1] < 0:
        print("Note: alpha1 < 0 → error correction active")
    else:
        print("Note: alpha1 ≥ 0 → weak/no error correction")
    
    plt.figure(figsize=(10,3.6))
    plt.plot(resid_e)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("ECM residuals")
    plt.tight_layout()
    plt.show()
else:
    print("\n[No ECM estimated — cointegration not found]")
    
# Pairwise cointegration tests

print("="*70)
print("COINTEGRATION TESTING: ALL PAIRS")
print("="*70)

GDP = data['GDP'].values.astype(float)
INFL = data['INFLATION'].values.astype(float)
IR = data['INTEREST_RATE'].values.astype(float)

CRIT_EG = -3.34
cointegration_results = []

pairs = [
    ("GDP", "INFLATION", GDP, INFL),
    ("GDP", "INTEREST_RATE", GDP, IR),
    ("INFLATION", "INTEREST_RATE", INFL, IR)
]

for var1_name, var2_name, var1, var2 in pairs:
    print(f"\n{'='*70}")
    print(f"Testing: {var1_name} ~ {var2_name}")
    print(f"{'='*70}")
    
    X_temp = np.column_stack((np.ones(len(var2)), var2))
    beta_temp, yhat_temp, uhat_temp, se_temp, r2_temp, _, _, n_temp, k_temp = ols_fit(var1, X_temp)
    
    print(f"\nStep 1: {var1_name} = β0 + β1·{var2_name} + u")
    print(f"  β0 = {beta_temp[0]:.6f}  (SE={se_temp[0]:.6f})")
    print(f"  β1 = {beta_temp[1]:.6f}  (SE={se_temp[1]:.6f})")
    print(f"  R² = {r2_temp:.4f}")
    
    delta_temp, tstat_temp = dickey_fuller_simple(uhat_temp)
    
    print(f"\nStep 2: DF test on residuals")
    print(f"  t-stat = {tstat_temp:.3f}  |  crit = {CRIT_EG}")
    
    is_cointegrated = tstat_temp < CRIT_EG
    decision_str = "✓ COINTEGRATED" if is_cointegrated else "✗ NOT cointegrated"
    print(f"  {decision_str}")
    
    cointegration_results.append({
        "Pair": f"{var1_name} ~ {var2_name}",
        "β0": f"{beta_temp[0]:.4f}",
        "β1": f"{beta_temp[1]:.4f}",
        "R²": f"{r2_temp:.4f}",
        "t-stat": f"{tstat_temp:.3f}",
        "Cointegrated?": "Yes" if is_cointegrated else "No"
    })

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

results_df = pd.DataFrame(cointegration_results)
print(results_df)
print(f"\nCritical value: {CRIT_EG} (5% significance)")

# Visualization
pair_labels = [r["Pair"] for r in cointegration_results]
t_stats = [float(r["t-stat"]) for r in cointegration_results]
r2_values = [float(r["R²"]) for r in cointegration_results]
is_coint = [r["Cointegrated?"] == "Yes" for r in cointegration_results]
colors = ['#27ae60' if coint else '#e74c3c' for coint in is_coint]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

# Plot 1: t-statistics
ax1.set_facecolor('#f8f9fa')
bars = ax1.barh(pair_labels, t_stats, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

ax1.axvline(CRIT_EG, color='#2c3e50', linewidth=3, linestyle='--', 
            label=f'Critical Value ({CRIT_EG})', alpha=0.8, zorder=1)
ax1.axvline(CRIT_EG, color='#34495e', linewidth=6, linestyle='--', alpha=0.2, zorder=0)

for bar, val in zip(bars, t_stats):
    offset = -0.3 if val < CRIT_EG else 0.3
    ha = 'right' if val < CRIT_EG else 'left'
    ax1.text(val + offset, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}', va='center', ha=ha, fontweight='bold', fontsize=10)

ax1.set_xlabel('t-statistic', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variable Pair', fontsize=12, fontweight='bold')
ax1.set_title('Engle-Granger Cointegration Tests', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10, framealpha=0.95, shadow=True)
ax1.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: R² values
ax2.set_facecolor('#f8f9fa')
bars2 = ax2.barh(pair_labels, r2_values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

for bar, val in zip(bars2, r2_values):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', ha='left', fontweight='bold', fontsize=10)

ax2.set_xlabel('R² (Goodness of Fit)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Variable Pair', fontsize=12, fontweight='bold')
ax2.set_title('Cointegrating Regression Fit', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlim(0, max(r2_values) * 1.15)
ax2.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', alpha=0.8, label='Cointegrated'),
    Patch(facecolor='#e74c3c', alpha=0.8, label='Not Cointegrated')
]
ax2.legend(handles=legend_elements, fontsize=10, framealpha=0.95, shadow=True)

plt.tight_layout()
plt.show()

# QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/
# QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/
# QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/QUESTION 4/

file_path = 'VU_MVE_assignment_panel_data.csv'

df = pd.read_csv(file_path)

#Task 4.1.1

#print(df)
#A dictionary to store the results in and a list to store the sum of squared residuals in
results = {}
ussr_list = []
#Loop to estimate OLS for each country
for c, g in df.groupby("country"):
    x = sm.add_constant(g[['Investment', 'Human_capital', 'Trade_openness']])
    y = g['GDP_growth']

    res = sm.OLS(y, x).fit()
    residuals = res.resid

    #results[c] = {"params": res.params, "residuals": res.resid}
    results[c] = res.params

    ussr_list.append(res.ssr)


ussr = sum(ussr_list)
#print(results)

#Task 4.1.2

#Transpose the results and store in preparation for visualisation
df_coefficients = pd.DataFrame(results).T

#print(df_coefficients)

#Histogram across countries

plt.figure()
plt.hist(df_coefficients['Investment'], bins=10, edgecolor='black', rwidth=0.8, color='#4C72B0')
plt.xlabel(r'$\hat{\beta}_{1,i}$ (Investment) across countries')
plt.ylabel('Frequency')
plt.title(r'Histogram of $\hat{\beta}_{1,i}$ (Investment) across countries')
plt.show()

#Box plot for each coefficient

plt.figure()
df_coefficients[['Investment', 'Human_capital', 'Trade_openness']].boxplot()
plt.ylabel('Coefficient')
plt.title(r'Box plots of $\hat{\beta}_{1,i}$, $\hat{\beta}_{2,i}$, $\hat{\beta}_{3,i}$')
plt.show()

#Scatter plot

plt.figure()
plt.scatter(df_coefficients['Investment'], df_coefficients['Human_capital'])
plt.xlabel(r'$\hat{\beta}_{1,i}$ (Investment)')
plt.ylabel(r'$\hat{\beta}_{2,i}$ (Human capital)')
plt.show()


# Task 4.2

# first we define x and y for the complete dataset
x_pooled = sm.add_constant(df[['Investment', 'Human_capital', 'Trade_openness']])
y_pooled = df['GDP_growth']

# Task 4.2.1

#now we fit the model
model_pooled = sm.OLS(y_pooled, x_pooled).fit()

# and we get the restricted Sum of squares
rssr = model_pooled.ssr
#print(rssr)


# Get pooled coefficients and standard errors for the table
pooled_params = model_pooled.params
pooled_bse = model_pooled.bse

# task 4.2.2

# mean group estimates
mg_params = df_coefficients.mean()

#sterror of mean group
mg_bse = df_coefficients.sem()

#Task 4.2.3 create the F-test
#first we calculate N, T and k
N = df['country'].nunique()
T = df['year'].nunique()

k = len(results[list(results.keys())[0]]) - 1

#print(f"N={N}, T={T}, k={k}")
# Then we calculate the degrees of freedom
df1 = k * (N-1)
df2 = N * (T - k - 1)

#Now we calculate the F-test
numerator = (rssr - ussr) / df1
denominator = ussr / df2

F_stat = numerator / denominator
p = 1 - f.cdf(F_stat, df1, df2)

print(f"Slope homogeneity F-test: Stat = {F_stat}, p-value = {p}")

# And we create the table

comparison_table = pd.DataFrame({'Pooled Coeff': pooled_params, 'Pooled SE': pooled_bse, 'MG Coeff': mg_params, 'MG SE': mg_bse})

print("\nComparison of Estimators:")
print(comparison_table)

#print(comparison_table.to_latex(float_format="%.4f"))

# QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/
# QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/
# QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/QUESTION 5/

# QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/
# QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/
# QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/QUESTION 6/

# Task 6.1

data = pd.read_csv('VU_MVE_assignment_panel_data.csv')
data.set_index(["country", "year"], inplace=True)

#print(data)

# First we need to calculate the mean per year

cross_sectional_avg = data.groupby('year').mean()
#print(cross_sectional_avg)
cross_sectional_avg.columns = [f"{col}_bar" for col in cross_sectional_avg.columns]

#print(cross_sectional_avg.columns)

# Merge these averages with the source data

data_cce = data.join(cross_sectional_avg, on='year')

#print(data_cce)

countries = data.index.get_level_values('country').unique()
#print(countries)

cce_coefs = []

for country in countries:
    #Slice data per country
    country_data = data_cce.xs(country, level='country')

    # Define the variables
    y = country_data['GDP_growth']

    x_vars = ['Investment', 'Human_capital', 'Trade_openness', 'GDP_growth_bar', 'Investment_bar', 'Human_capital_bar', 'Trade_openness_bar']
    x = country_data[x_vars]

    # Add a constant for alpha_i

    x = sm.add_constant(x)

    # Use statsmodels to fit the OLS model

    model = sm.OLS(y, x).fit()

    # Extract coefficients for original independent variables (investment, Human_capital and Trade_openness)
    beta_i = model.params[['Investment', 'Human_capital', 'Trade_openness']]

    cce_coefs.append(beta_i)

#print(cce_coefs)

cce_coefs_df = pd.DataFrame(cce_coefs, index=countries)

#print(cce_coefs_df)

# Now we can calculate the Mean Group Coefficients (averages of the beta_i's) and the standard errors
# Standard error of Mean Group Estimator is calculated as the standard error of the mean of individual coefficients

cce_mgc_params = cce_coefs_df.mean()

n = len(countries)
cce_mgc_se = cce_coefs_df.std()/ np.sqrt(n)

print(f"CCE Mean Group Estimates: {cce_mgc_params}")
print(f"\nCCE Mean Group Standard Errors: {cce_mgc_se}")

#task 6.2.1

#prepare pooled OLS and Mean Group data and CCE
df_pooled_res = pd.DataFrame({'Pooled OLS': pooled_params, 'Pooled SE': pooled_bse})

df_mg_res = pd.DataFrame({'Mean Group': mg_params, 'MG SE': mg_bse})

df_cce_res = pd.DataFrame({'CCE MG': cce_mgc_params, 'CCE MG SE': cce_mgc_se})

comparison = pd.concat([df_pooled_res, df_mg_res, df_cce_res], axis=1)

order = ['const', 'Investment', 'Human_capital', 'Trade_openness']

final_order = [idx for idx in order if idx in comparison.index]
comparison = comparison.reindex(final_order)

print(comparison)

#task 6.2.2

# First we have to identify outliers
investment_mean = cce_coefs_df['Investment'].mean()
investment_std = cce_coefs_df['Investment'].std()

threshold = 2
outliers = cce_coefs_df[np.abs(cce_coefs_df['Investment'] - investment_mean) > threshold *investment_std]

print(f"Outliers: {outliers}")

#Now we re-estimate the CCE and drop the outliers
cce_robust = cce_coefs_df.drop(outliers.index)

# And we calculate the new CCE
cce_mg_robust_params = cce_robust.mean()
cce_mg_robust_se = cce_robust.std()/ np.sqrt(len(cce_robust))

# We calculate the sensitivity (pct of change)
sensitivity = ((cce_mg_robust_params - cce_mgc_params) / cce_mgc_params) * 100

print(f"CCE Mean Group after removing outliers:")
print(f"\n{cce_mg_robust_params}")

print(f"\nSensitivity: {sensitivity}")

robustness_table = pd.DataFrame({'Original': cce_mgc_params, 'Robust (without outliers)': cce_mg_robust_params, 'Sensitivity': sensitivity})