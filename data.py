"""Sample datasets for the AutoML demo (simulated)."""
import numpy as np
import pandas as pd


def customer_churn(n=1500, seed=1):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"tenure_months": r.integers(1, 72, n), "monthly_charges": r.normal(70, 25, n).round(2),
                       "contract": r.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2]),
                       "support_calls": r.poisson(1.5, n), "online_billing": r.choice(["Yes", "No"], n)})
    logit = -1 - 0.04 * df.tenure_months + 0.02 * df.monthly_charges + 0.5 * df.support_calls + 1.2 * (df.contract == "Month-to-month")
    df["churn"] = np.where(r.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df.loc[r.random(n) < .03, "monthly_charges"] = np.nan
    return df


def energy_consumption(n=1500, seed=2):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"floor_area_m2": r.normal(150, 50, n).clip(40).round(), "occupants": r.integers(1, 7, n),
                       "outdoor_temp_c": r.normal(5, 12, n).round(1), "insulation": r.choice(["Poor", "Average", "Good"], n),
                       "heat_pump": r.choice([0, 1], n)})
    df["kwh_per_month"] = (300 + 4 * df.floor_area_m2 + 60 * df.occupants - 18 * df.outdoor_temp_c
                           + df.insulation.map({"Poor": 250, "Average": 100, "Good": 0}) - 180 * df.heat_pump
                           + r.normal(0, 60, n)).round()
    return df


SAMPLES = {"Customer churn (classification)": (customer_churn, "churn"),
           "Energy consumption (regression)": (energy_consumption, "kwh_per_month")}
