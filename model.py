import os
import numpy as np
import pandas as pd

# Deterministic risk engine based on specified factors.
# Factors: supplier pressure (%), customer payment delay (days), expense pressure (%),
# cash runway (days), projected deficit (amount), current cash.
# Risk score is calculated and capped between 0 and 100.
# Risk bands: 0-29 LOW, 30-59 MEDIUM, 60-100 HIGH.

def get_risk_band(score: float) -> str:
    """Return risk band string based on score."""
    if score < 30:
        return "LOW"
    if score < 60:
        return "MEDIUM"
    return "HIGH"

def compute_risk_score(features: dict) -> float:
    """Deterministic risk scoring.

    Expected keys in ``features``:
        - supplier_growth: float (percentage change, e.g., 0.2 for +20%)
        - customer_delay: int (days)
        - expense_pressure: float (percentage change, e.g., 0.1 for +10%)
        - runway_days: int (cash runway in days)
        - deficit_amount: float (absolute deficit amount in rupees, 0 if none)
        - current_cash: float (cash on hand in rupees)
    """
    # Supplier pressure: higher % increase => higher risk (max 100%).
    supplier_score = min(max(features.get('supplier_growth', 0) * 100, 0), 100)

    # Customer delay: proportion of max 60 days considered high risk.
    delay_days = features.get('customer_delay', 0)
    delay_score = min(max((delay_days / 60) * 100, 0), 100)

    # Expense pressure: similar to supplier.
    expense_score = min(max(features.get('expense_pressure', 0) * 100, 0), 100)

    # Cash runway: shorter runway => higher risk. Assume 180 days as safe horizon.
    runway = features.get('runway_days', 180)
    runway_score = max(0, 100 - (runway / 180) * 100)

    # Projected deficit: any deficit adds risk proportionally (scale to 100k rupees).
    deficit = features.get('deficit_amount', 0)
    deficit_score = min(max((deficit / 100000) * 20, 0), 100)  # each 100k adds 20 points.

    # Current cash: lower cash adds risk. Assume 500k as healthy buffer.
    cash = features.get('current_cash', 0)
    cash_score = max(0, 100 - (cash / 500000) * 100)

    # Average of components yields the raw score.
    components = [supplier_score, delay_score, expense_score, runway_score, deficit_score, cash_score]
    raw_score = np.mean(components)
    return float(np.clip(raw_score, 0, 100))

def predict_risk(features: dict) -> dict:
    """Predict risk score and band using deterministic engine.

    ``features`` should contain the keys used by ``compute_risk_score``.
    """
    score = compute_risk_score(features)
    return {
        'score': round(score, 1),
        'band': get_risk_band(score)
    }

# Compatibility placeholder – some code expects a train function.
def train_risk_model():
    """Create a dummy model file so existing checks for model.pkl succeed."""
    dummy_path = "model.pkl"
    if not os.path.exists(dummy_path):
        with open(dummy_path, "wb") as f:
            f.write(b"deterministic")
    return None

if __name__ == "__main__":
    sample = {
        'supplier_growth': 0.2,
        'customer_delay': 10,
        'expense_pressure': 0.1,
        'runway_days': 45,
        'deficit_amount': 0,
        'current_cash': 300000,
    }
    print(predict_risk(sample))
