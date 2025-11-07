# Code documentation

Please briefly report here notes for the parts of your code that need to be used by the others,
under a specific section.

- [Code documentation](#code-documentation)
  - [helper.py](#helperpy)
    - [Function: `acf(x)`](#function-acfx)
    - [Function: `dickey_fuller(x)`](#function-dickey_fullerx)
    - [Function: `ljung_box(x)`](#function-ljung_boxx)
  - [Part 1, Question 2](#part-1-question-2)
    - [Additional notes](#additional-notes)

## helper.py
### Function: `acf(x)`
Computes the **autocorrelation function** of a time series.

**Parameters**
- `x`: `np.ndarray` — input time series.

**Returns**
- `np.ndarray`: autocorrelation values (biased version).

### Function: `dickey_fuller(x)`
Performs the **Dickey–Fuller test** for a unit root.

**Parameters**
- `x`: `np.ndarray` — time series.

**Returns**
- `(delta_hat, t_stat)`: regression coefficient and t-statistic.

**Model**
Δxₜ = δxₜ₋₁ + uₜ

Reject the null of a unit root if the t-statistic is **below** the critical value.

---

### Function: `ljung_box(x)`
Computes the **Ljung–Box Q-statistic** and p-values for serial correlation.

**Parameters**
- `x`: `np.ndarray` — time series.

**Returns**
- `(lj_bx, pv)`: test statistic and p-values.

## Part 1, Question 2

This section is in the <code>p1_q2.py</code> file.

Important variables:

- <code>correlated_random_walk</code> -> the serially correlated series to be generated $z_t$
- <code>correlated_errors</code> -> the stationary ar(1) errors $u_t$
- <code>innovation</code> -> the iid standard normal exogenous components

These may be imported in other files if needed later in the code. Say we need to import <code>correlated_random_walk</code>, then we would use:

<code>from p1_q2 import correlated_random_walk </code>

Note that this launches the entire file, so some conditionals inside it prevent unwanted code chunks to also run (e.g. plots).

This is done using the <code>ismain</code> variable defined as:

```python
ismain = __name__ == '__main__'
```

which is true only when the code is runned from the file itself.

### Additional notes

- the long run and contemporaneous variance are the same because the errors are stationary