import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed = 123)
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
first_difference_centered = first_difference - np.mean(first_difference)
len_series = len(first_difference)
acf_biased = np.correlate(first_difference_centered, first_difference_centered, mode='full')[len_series-1:]
lags = np.arange(len_series)
acf_nonnormalized = acf_biased/(len_series - lags)
acf = acf_nonnormalized / acf_nonnormalized[0]

if ismain:
    plt.bar(np.arange(len(acf)), acf)
    plt.xlabel("Lag")
    plt.ylabel("Unbiased ACF")
    plt.title(r"Autocorrelation of $\Delta z_t$")
    plt.show()

error_variance = innovation_variance/(1.0 - error_ar_coefficient)**2
if ismain: print(f"Contemporaneous and long run variance are: {error_variance}")
