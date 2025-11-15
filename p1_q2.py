import numpy as np
import matplotlib.pyplot as plt

from helper import acf, dickey_fuller, ljung_box, aug_dickey_fuller

rng: np.random.Generator = np.random.default_rng(seed = 123)
ismain: bool = __name__ == '__main__'

# Task 2.1

innovation_variance = 1.0
innovation = rng.normal(loc=0.0, scale=innovation_variance, size=550)
error_ar_coefficient = 0.5

correlated_errors = np.convolve(innovation, error_ar_coefficient**np.arange(550), mode='full')[:550]
correlated_random_walk = np.cumsum(correlated_errors)[50:]

if ismain:
    plt.plot(correlated_random_walk)
    plt.xlabel("Time")
    plt.ylabel(r"$z_t$")
    plt.title("Random Walk with serially correlated innovations")
    plt.show()

first_difference = np.diff(correlated_random_walk)
len_series = len(first_difference)
lags = np.arange(len_series)

acf_crw = acf(first_difference)

if ismain:
    plt.bar(lags, acf_crw)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(r"Autocorrelation of $\Delta z_t$")
    plt.show()

cont_variance = innovation_variance/(1.0 - error_ar_coefficient**2) # see report for derivation
if ismain: 
    print(f"Contemporaneous variance: {cont_variance}")
    print(f"Long run variance: {3 * cont_variance}") # see report for derivation

# Task 2.2
df_coeff, df_stat = dickey_fuller(correlated_random_walk)
if ismain:
    print(f"""Dickey Fuller estimated coefficients {df_coeff} and t-statistic {df_stat} \n
    Against the critical values 1% -2.58, 5% -1.95, 10% -1.62 \n,
    Since {df_stat.__round__(2)} > -1.95 we fail to reject the null hypothesis.
    """)

df_residuals = first_difference - df_coeff * correlated_random_walk[:-1]
lb_stat, lb_pv = ljung_box(df_residuals)
failed = np.sum([lb_pv <= 0.05])

if ismain: print(f"Number of failed Ljung Box tests {failed} on {len(lb_pv)} total")

# Task 2.3

def calc_aic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0: return np.inf
    return 2 * k_params + n_obs * np.log(ssr / n_obs)

def calc_bic(n_obs, ssr, k_params):
    if ssr <= 0 or n_obs <= 0: return np.inf
    return k_params * np.log(n_obs) + n_obs * np.log(ssr / n_obs)

if ismain:
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
    plt.bar(range(len(acf_adf_residuals)), acf_adf_residuals)
    plt.xlabel("Lag")
    plt.ylabel("Unbiased ACF")
    plt.title(f"ACF of Residuals from ADF(k={best_k})")
    plt.show()

    lb_stat_adf, lb_pv_adf = ljung_box(adf_residuals)
    failed_adf = np.sum(lb_pv_adf <= 0.05)

    print(f"Ljung-Box Test: {failed_adf} out of {len(lb_pv_adf)} tests failed (p<=0.05).")
    print(f"Since we failed {failed_adf} out of {len(lb_pv_adf)}, we have not successfully removed serial correlation.")



# Task 2.4

if ismain:

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

if ismain:
    se_delta = df_coeff / df_stat
    tstatpp = (df_stat * (sigma_0_sq**0.5 / sigma_u_sq**0.5)) - ((sigma_u_sq - sigma_0_sq) * T * se_delta)/ ( 2* (sigma_u_sq**0.5)*(sigma_0_sq**0.5))

    print(f"\nPP corrected t statistic: {tstatpp}")
