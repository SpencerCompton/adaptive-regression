This repository contains estimators for linear regression we studied in [Attainability of Two-Point Testing Rates for Finite-Sample Location Estimation](https://arxiv.org/abs/2502.05730) by Spencer Compton and Gregory Valiant. See this paper for further details.

The two linear regression estimators are:
- `AdaptiveLqRegression`, which selects among $L_q$ regression estimates.
- `AutoRBDescRegressor`, which applies residual-balance descent with automatic whitening and radius selection for mean-zero heteroskedastic errors.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Example

```python
import numpy as np

from adaptive_regression import AdaptiveLqRegression, AutoRBDescRegressor

rng = np.random.default_rng(0)
n, d = 5000, 3

beta = np.array([1.0, -0.5, 0.25])
noise_names = ["N(0,1)", "Uniform [-1,1]"]
noises = [rng.normal(size=n), rng.uniform(-1, 1, size=n)]

print("true:       ", beta)

for i in range(len(noises)):
    X = rng.normal(size=(n, d))
    y = X @ beta + noises[i]
    adaptive_lq = AdaptiveLqRegression(random_state=0).fit(X, y)
    auto_rbdesc = AutoRBDescRegressor().fit(X, y)
    print("Noise:",noise_names[i])
    print("AdaptiveLq:", adaptive_lq.coef_,"error:",np.linalg.norm(adaptive_lq.coef_-beta))
    print("AutoRBDesc:", auto_rbdesc.coef_,"error:",np.linalg.norm(auto_rbdesc.coef_-beta))

```

The estimators use the columns supplied in `X` and do not add an intercept automatically. To fit an intercept, include a column of ones in `X`; its coefficient is then included normally in the returned coefficient vector.

Run the included example with:

```bash
python example.py
```
