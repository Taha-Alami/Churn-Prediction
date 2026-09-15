"""Synthetic customer data generator for a telecom-style subscription business.

This models a generic subscription telecom service, not any real company.
There is no real customer, billing, or network data anywhere in this
repository or its generating logic. This module exists so the rest of the
pipeline (feature engineering, CV training, calibration, SHAP, MLflow
logging) can be developed and evaluated end to end without access to any
real company's data warehouse.

Data-generating process (DGP)
------------------------------
Everything is driven by a single seeded ``numpy.random.default_rng(seed)``
so the dataset is fully deterministic for a given seed. The steps are:

1. **Demographics / account shape** (independent draws):
   - ``is_senior``: Bernoulli(p=0.16).
   - ``contract_type``: categorical over {month_to_month, one_year,
     two_year} with weights (0.55, 0.25, 0.20), reflecting that
     month-to-month plans are the modal contract in most subscription
     telecom books.
   - ``payment_method``: categorical over {electronic_check, mailed_check,
     bank_transfer, credit_card} with weights (0.35, 0.15, 0.25, 0.25).
   - ``autopay``: Bernoulli, with p=0.75 if ``payment_method`` is
     bank_transfer or credit_card, else p=0.20 (autopay is only easy to
     set up for those two payment rails).
   - ``has_multiple_lines``: Bernoulli(p=0.40).

2. **Tenure**: drawn from an exponential distribution (mean 24 months),
   clipped to [0, 72], truncated to integers. This produces the
   right-skewed tenure distribution typical of subscription businesses
   (many recent joiners, a long tail of loyal customers).

3. **Usage and service**:
   - ``avg_monthly_usage_gb``: lognormal, with the underlying mean shifted
     up for customers with multiple lines.
   - ``num_support_tickets_90d``: Poisson with lambda=0.6, plus an extra
     Poisson(0.9) draw for customers in their first 6 months (new
     customers generate more onboarding-related support contacts).

4. **Pricing**:
   - ``monthly_charges``: a base charge plus add-ons for multiple lines and
     higher usage, plus Gaussian noise, clipped to a plausible [20, 150]
     range.
   - ``total_charges``: approximately ``monthly_charges * tenure_months``
     with multiplicative noise, representing lifetime billing. Customers
     with ``tenure_months == 0`` (i.e. joined this billing cycle) get a
     null ``total_charges`` -- mirroring a well-known real-world quirk in
     telecom billing extracts where a brand-new account has no completed
     billing cycle yet.

5. **Churn label**: generated from a logistic model over standardized
   features so the signal is real but not trivially separable:

   ``logit = -2.5
       - 0.045 * tenure_months
       + 1.35  * I(contract_type == month_to_month)
       + 0.55  * I(contract_type == one_year)
       + 0.40  * num_support_tickets_90d
       - 0.35  * avg_monthly_usage_gb_z
       - 0.90  * autopay
       + 0.015 * monthly_charges
       + 0.20  * is_senior
       + 1.10  * I(contract_type == month_to_month AND num_support_tickets_90d >= 2)
       + 0.90  * I(tenure_months < 3)
       + noise ~ Normal(0, 0.75)``

   ``churn_probability = sigmoid(logit)``, and ``churn`` is a Bernoulli
   draw from that probability. The coefficients were chosen by trial and
   error to land the overall churn rate in the 20-30% range typically
   reported for telecom subscriber bases, while keeping every listed
   driver directionally realistic: short tenure, month-to-month contracts,
   more support tickets, low usage, no autopay, and higher bills all push
   churn probability up. The last two terms are deliberate non-additive
   effects -- an interaction (frustrated month-to-month customers with
   multiple support tickets churn disproportionately more than the sum of
   the two individual effects) and a threshold "onboarding cliff" for
   customers in their first two months -- so that a plain linear model
   cannot fully capture the true decision boundary and a model that can
   learn interactions (like the gradient-boosted tree used as the main
   model here) has genuine, non-fabricated room to outperform it.

This is a deliberately simple DGP. It is documented in full here so it is
unambiguous that model performance numbers reported elsewhere in this repo
come from a known, inspectable signal -- not from real subscriber behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ("month_to_month", "one_year", "two_year")
CONTRACT_WEIGHTS = (0.55, 0.25, 0.20)
PAYMENT_METHODS = ("electronic_check", "mailed_check", "bank_transfer", "credit_card")
PAYMENT_WEIGHTS = (0.35, 0.15, 0.25, 0.25)

COLUMNS = [
    "customer_id",
    "tenure_months",
    "contract_type",
    "monthly_charges",
    "total_charges",
    "payment_method",
    "autopay",
    "num_support_tickets_90d",
    "avg_monthly_usage_gb",
    "has_multiple_lines",
    "is_senior",
    "churn",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_customers(n_customers: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic, telecom-style subscription customer table.

    Parameters
    ----------
    n_customers:
        Number of customer rows to generate.
    seed:
        Seed for ``numpy.random.default_rng``. Same seed + same
        ``n_customers`` always produces an identical dataframe.

    Returns
    -------
    pandas.DataFrame with the columns documented in the module docstring.
    """
    rng = np.random.default_rng(seed)
    n = n_customers

    is_senior = rng.binomial(1, 0.16, size=n).astype(bool)
    contract_type = rng.choice(CONTRACT_TYPES, size=n, p=CONTRACT_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_WEIGHTS)

    autopay_base_p = np.where(np.isin(payment_method, ["bank_transfer", "credit_card"]), 0.75, 0.20)
    autopay = rng.binomial(1, autopay_base_p).astype(bool)

    has_multiple_lines = rng.binomial(1, 0.40, size=n).astype(bool)

    tenure_months = np.clip(rng.exponential(scale=24.0, size=n), 0, 72).astype(int)

    usage_mean = np.where(has_multiple_lines, 3.6, 3.1)
    avg_monthly_usage_gb = rng.lognormal(mean=usage_mean, sigma=0.5, size=n)
    avg_monthly_usage_gb = np.clip(avg_monthly_usage_gb, 1, 500)

    tickets_base = rng.poisson(0.6, size=n)
    new_customer_bonus = np.where(tenure_months < 6, rng.poisson(0.9, size=n), 0)
    num_support_tickets_90d = tickets_base + new_customer_bonus

    monthly_charges = (
        35.0 + 18.0 * has_multiple_lines + 0.15 * avg_monthly_usage_gb + rng.normal(0, 8.0, size=n)
    )
    monthly_charges = np.clip(monthly_charges, 20.0, 150.0)

    billing_noise = rng.normal(1.0, 0.05, size=n)
    total_charges = monthly_charges * tenure_months * billing_noise
    total_charges = np.where(tenure_months == 0, np.nan, total_charges)

    usage_z = (avg_monthly_usage_gb - avg_monthly_usage_gb.mean()) / avg_monthly_usage_gb.std()

    mtm_high_tickets = (contract_type == "month_to_month") & (num_support_tickets_90d >= 2)
    onboarding_cliff = tenure_months < 3

    logit = (
        -2.5
        - 0.045 * tenure_months
        + 1.35 * (contract_type == "month_to_month")
        + 0.55 * (contract_type == "one_year")
        + 0.40 * num_support_tickets_90d
        - 0.35 * usage_z
        - 0.90 * autopay
        + 0.015 * monthly_charges
        + 0.20 * is_senior
        + 1.10 * mtm_high_tickets
        + 0.90 * onboarding_cliff
        + rng.normal(0, 0.75, size=n)
    )
    churn_probability = _sigmoid(logit)
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:07d}" for i in range(n)],
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": np.round(total_charges, 2),
            "payment_method": payment_method,
            "autopay": autopay,
            "num_support_tickets_90d": num_support_tickets_90d,
            "avg_monthly_usage_gb": np.round(avg_monthly_usage_gb, 2),
            "has_multiple_lines": has_multiple_lines,
            "is_senior": is_senior,
            "churn": churn,
        }
    )
    return df[COLUMNS]


def main() -> None:
    """CLI entry point: generate the dataset and write it to data/raw/customers.parquet."""
    from churn.config import settings

    df = generate_synthetic_customers(n_customers=settings.n_customers, seed=settings.random_seed)
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.raw_data_path, index=False)
    print(f"Wrote {len(df)} rows to {settings.raw_data_path}")
    print(f"Churn rate: {df['churn'].mean():.3%}")


if __name__ == "__main__":
    main()
