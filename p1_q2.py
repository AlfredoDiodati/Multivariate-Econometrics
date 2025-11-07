import numpy as np
import matplotlib.pyplot as plt

from helper import acf, dickey_fuller, ljung_box

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
    plt.ylabel("Unbiased ACF")
    plt.title(r"Autocorrelation of $\Delta z_t$")
    plt.show()

error_variance = innovation_variance/(1.0 - error_ar_coefficient)**2
if ismain: print(f"Contemporaneous and long run variance are: {error_variance}")

# Task 2.2
df_coeff, df_stat = dickey_fuller(correlated_random_walk)

if ismain:
    print(f"""Dickey Fuller estimated coefficients {df_coeff} and t-statistic {df_stat} \n
    Against the critical values 1% -2.58, 5% -1.95, 10% -1.62 \n,
    Since {df_stat} > -1.95 we fail to reject the null hypothesis.
    """)
    
df_residuals = first_difference - df_coeff * correlated_random_walk[:-1]
lb_stat, lb_pv = ljung_box(df_residuals)
failed = np.sum([lb_pv <= 0.05])

if ismain: print(f"Number of failed Ljung Box tests {failed} on {len(lb_pv)} total")