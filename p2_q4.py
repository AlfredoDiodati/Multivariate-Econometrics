import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy.distutils.system_info import numarray_info
from scipy.stats import f

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