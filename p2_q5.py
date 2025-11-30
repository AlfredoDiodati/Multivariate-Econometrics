import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

ismain = __name__ == '__main__' 

# Task 5.1
data = pd.read_csv('VU_MVE_assignment_panel_data.csv')
data.set_index(["country", "year"], inplace=True)

fixed_effect_model = PanelOLS.from_formula(
    "GDP_growth ~ Investment + Human_capital + Trade_openness + EntityEffects",
    data=data
)
fixed_effect_model = fixed_effect_model.fit()

if ismain: print(fixed_effect_model.summary)

residuals = fixed_effect_model.resids

def pairwise_residual_correlation(residuals, country_i, country_j):
    """
    Computes pairwise residual correlation
    Parameters
    residuals : pd.Series
        MultiIndex Series with index = (country, year)
    country_i, country_j : str
        Names of the two countries to compare.
    """
    
    e_i = residuals.xs(country_i, level="country")
    e_j = residuals.xs(country_j, level="country")
    
    e_i, e_j = e_i.align(e_j, join="inner")
    
    num = np.sum(e_i * e_j)
    den = np.sqrt(np.sum(e_i**2) * np.sum(e_j**2))
    
    return num / den

def cd_test_statistic(residuals) ->tuple[float, float]:
    """
    Computes the Pesaran CD statistic
    Input
    residuals : pd.Series
        MultiIndex Series with index (country, year)
    """
    
    countries = residuals.index.get_level_values("country").unique()
    N = len(countries)

    T = len(residuals.xs(countries[0], level="country"))
    sum_rho = 0.0
    for idx_i in range(N - 1):
        ci = countries[idx_i]
        for idx_j in range(idx_i + 1, N):
            cj = countries[idx_j]
            rho_ij = pairwise_residual_correlation(residuals, ci, cj)
            sum_rho += rho_ij

    CD = np.sqrt(2 * T / (N * (N - 1))) * sum_rho
    p_value = 2 * (1 - norm.cdf(abs(CD)))
    return CD, p_value

CD, p_value = cd_test_statistic(residuals)

if ismain: print(f"CD statistics {CD.__round__(3)} and p-value {p_value.__round__(3)}")

# Task 5.2

def residual_correlation_matrix(residuals):
    countries = residuals.index.get_level_values("country").unique()
    C = pd.DataFrame(index=countries, columns=countries, dtype=float)
    
    for i in countries:
        e_i = residuals.xs(i, level="country")
        denom_i = np.sqrt(np.sum(e_i**2))
        for j in countries:
            e_j = residuals.xs(j, level="country")
            denom_j = np.sqrt(np.sum(e_j**2))
            e_i2, e_j2 = e_i.align(e_j, join="inner")
            num = np.sum(e_i2 * e_j2)

            C.loc[i, j] = num / (denom_i * denom_j)
    return C

# Heatmap is not clearly readible with full names, so we map them to abbreviations

mapping = {
    "Germany": "DE",
    "France": "FR",
    "Italy": "IT",
    "Spain": "ES",
    "Poland": "PL",
    "Romania": "RO",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Greece": "GR",
    "Portugal": "PT",
    "Czech Republic": "CZ",
    "Hungary": "HU",
    "Sweden": "SE",
    "Austria": "AT",
    "Bulgaria": "BG",
    "Denmark": "DK",
    "Finland": "FI",
    "Slovakia": "SK",
    "Ireland": "IE",
    "Croatia": "HR",
    "Lithuania": "LT",
    "Slovenia": "SI",
    "Latvia": "LV",
    "Estonia": "EE",
    "Cyprus": "CY",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Norway": "NO",
    "Switzerland": "CH",
    "Iceland": "IS",
}

cd_matrix = residual_correlation_matrix(residuals)
cd_matrix = cd_matrix.rename(index=mapping, columns=mapping)

if ismain:
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cd_matrix,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        annot_kws={"size": 4},
        cbar_kws={"shrink": 0.6}
    )

    plt.xticks(rotation=90, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()