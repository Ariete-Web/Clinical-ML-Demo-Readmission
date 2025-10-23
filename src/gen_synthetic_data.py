feat: data generator
# Makes data/synthetic_readmission.csv (no downloads needed)
import numpy as np, pandas as pd
from pathlib import Path

def gen_readmission(N=25000, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=N)
    sex = rng.choice(['F','M'], size=N)
    num_chronic = rng.poisson(1.5, size=N)
    days_in_hosp = rng.poisson(3 + 0.02*(age-50).clip(0), size=N)
    prior_admits = rng.poisson(0.5 + 0.2*(num_chronic>2), size=N)
    z = -2.2 + 0.015*(age-50) + 0.25*(sex=='M') + 0.3*np.log1p(num_chronic) \
        + 0.08*days_in_hosp + 0.4*np.log1p(prior_admits)
    p = 1/(1+np.exp(-z))
    y = (rng.random(N) < p).astype(int)
    return pd.DataFrame({
        'age': age, 'sex': sex, 'num_chronic': num_chronic,
        'days_in_hosp': days_in_hosp, 'prior_admits': prior_admits,
        'readmitted': y
    })

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = gen_readmission()
    df.to_csv("data/synthetic_readmission.csv", index=False)
    print("Wrote data/synthetic_readmission.csv with", len(df), "rows")
