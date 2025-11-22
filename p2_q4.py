import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

file_path = 'VU_MVE_assignment_panel_data.csv'

df = pd.read_csv(file_path)

#Task 4.1.1

#print(df)
#A dictionary to store the results in
results = {}
#Loop to estimate OLS for each country
for c, g in df.groupby("country"):
    x = sm.add_constant(g[['Investment', 'Human_capital', 'Trade_openness']])
    y = g['GDP_growth']

    res = sm.OLS(y, x).fit()
    residuals = res.resid

    #results[c] = {"params": res.params, "residuals": res.resid}
    results[c] = res.params


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