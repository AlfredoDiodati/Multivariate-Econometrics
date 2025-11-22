import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from helper import acf, ar_coefficients, ar_standard_errors, ljung_box, dickey_fuller

ismain = __name__ == '__main__'
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

if ismain:
    plt.plot(random_walk, label="Random Walk")
    plt.plot(ar, label=rf"AR(1) $\rho = {ar_coefficient}$")
    plt.xlabel("Time")
    plt.ylabel(r"Value")
    plt.title("Random Walk and AR(1) processes")
    plt.legend()
    plt.show()
    
acf_rw = acf(random_walk)
acf_ar = acf(ar)
lags = np.arange(acf_rw.shape[0])

if ismain:
    plt.bar(lags, acf_rw, label="Random Walk")
    plt.bar(lags, acf_ar, label=rf"AR(1) $\rho = {ar_coefficient}$")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(r"Autocorrelation of $y_t$ and $x_t$")
    plt.legend()
    plt.show()
    
# task 1.2

intercept, slope = ar_coefficients(random_walk)
se_intercept, se_slope = ar_standard_errors(random_walk)
t_intercept = intercept / se_intercept
t_slope = slope / se_slope
pv_intercept = 2 * (1 - t.cdf(abs(t_intercept), df= T - 2))
pv_slope = 2 * (1 - t.cdf(abs(t_slope), df= T - 2))

if ismain:
    print(f"Estimated intercept {intercept.__round__(3)} with standard error {se_intercept.__round__(3)}")
    print(f"Estimated slope {slope.__round__(3)} with standard error {se_slope.__round__(3)}")
    print(f"Intercept and Slope t statistics: {t_intercept.__round__(3)} and {t_slope.__round__(3)} respectively")
    print(f"...and corresponding p.values: {pv_intercept.__round__(3)} and {pv_intercept.__round__(3)} respectively")
    
residuals = random_walk[1:] - intercept - slope * random_walk[:-1]
R2 = 1.0 - np.sum(residuals**2)/np.sum((random_walk[1:] - np.mean(random_walk[1:]))**2)
acf_residuals = acf(residuals)

if ismain:
    
    print(fr"\n AR(1) with intercept's R^2: {R2.__round__(3)}")
    
    plt.bar(lags[:-1], acf_residuals)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("ACF of Mispecified AR(1) with intercept")
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

if ismain:
    print(f"Ljung-Box test statistics: \n{lb_table}")
    print(f"\nLjung-Box test statistics: \n{pv_table}")
    
if ismain:
    plt.plot(residuals)
    plt.xlabel("Time")
    plt.ylabel(r"Value")
    plt.title("Residuals of mispecified AR(1) with intercept")
    plt.show()
    
# Task 1.3
_, df_stat = dickey_fuller(random_walk)
df_critical_v = -1.62
isrejected = df_stat < df_critical_v

if ismain:
    print(f"\n Dickey Fuller statistics of Random Walk {df_stat.__round__(3)}")
    if isrejected: print(f"Since {df_stat} < {df_critical_v} we reject the null at a 10% confidence interval")
    else: print(f"Since {df_stat.__round__(3)} > {df_critical_v} we fail to reject the null at a 10% confidence interval")
    
first_differences = random_walk[1:]-random_walk[:-1]
df_coeff_diff, df_stat_diff = dickey_fuller(first_differences)

if ismain:
    print(f"\n First differences of random walk \nDF coefficient {df_coeff_diff.__round__(3)} and statistics {df_stat_diff.__round__(3)}")
    print(f"Therefore the null hpt. of a unit root is rejected, and we conclude the random walk is I(1) \n")
    
df_coeff_ar, df_stat_ar = dickey_fuller(ar)
if ismain:
    print(f"Stationary AR(1) \n DF coefficient {df_coeff_ar.__round__(3)} and statistics {df_stat_ar.__round__(3)}")
    print(f"Therefore the null hpt. of a unit root is rejected.")