import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import p2_q4


ismain = __name__ == '__main__'

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
df_pooled_res = pd.DataFrame({'Pooled OLS': p2_q4.pooled_params, 'Pooled SE': p2_q4.pooled_bse})

df_mg_res = pd.DataFrame({'Mean Group': p2_q4.mg_params, 'MG SE': p2_q4.mg_bse})

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