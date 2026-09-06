import pandas as pd
import numpy as np
from datetime import timedelta, date
import os

def generate_msme_data(history_days=180, future_days=60, output_dir="data"):
    np.random.seed(42)
    
    total_days = history_days + future_days
    start_date = date.today() - timedelta(days=history_days)
    
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    
    # Initialize lists
    customer_payments_in = np.zeros(total_days)
    supplier_payments = np.zeros(total_days)
    payroll = np.zeros(total_days)
    rent = np.zeros(total_days)
    utilities = np.zeros(total_days)
    loan_emi = np.zeros(total_days)
    other_expenses = np.zeros(total_days)
    
    base_revenue = 10000
    base_supplier_cost = 6000
    
    for i, current_date in enumerate(dates):
        # Weekly seasonality + random noise for revenue
        day_of_week = current_date.weekday()
        
        # Weekends have lower revenue
        if day_of_week >= 5:
            daily_rev = np.random.normal(base_revenue * 0.3, 1000)
        else:
            daily_rev = np.random.normal(base_revenue, 2000)
            
        # Simulate occasional late payments (bumpy revenue)
        if np.random.random() < 0.1:  # 10% chance of a big payment arriving
            daily_rev += np.random.normal(base_revenue * 2, 2000)
        if np.random.random() < 0.15: # 15% chance of low revenue (late payment)
            daily_rev *= 0.2
            
        customer_payments_in[i] = max(0, daily_rev)
        
        # Supplier payments (every week or random spikes)
        if day_of_week == 2: # Pay suppliers on Wednesdays
            supplier_payments[i] = max(0, np.random.normal(base_supplier_cost * 5, 2000))
        elif np.random.random() < 0.05:
            supplier_payments[i] = max(0, np.random.normal(base_supplier_cost * 2, 1000))
            
        # Payroll every 2 weeks (1st and 15th)
        if current_date.day in [1, 15]:
            payroll[i] = 25000
            
        # Rent on the 1st
        if current_date.day == 1:
            rent[i] = 15000
            
        # Utilities on the 5th
        if current_date.day == 5:
            utilities[i] = 3000
            
        # Loan EMI on the 10th
        if current_date.day == 10:
            loan_emi[i] = 12000
            
        # Other expenses (small daily)
        other_expenses[i] = max(0, np.random.normal(500, 200))
        
    # Calculate balances
    opening_balance = np.zeros(total_days)
    closing_balance = np.zeros(total_days)
    
    # Starting balance
    current_cash = 100000
    
    for i in range(total_days):
        opening_balance[i] = current_cash
        
        closing_cash = (
            opening_balance[i] +
            customer_payments_in[i] -
            supplier_payments[i] -
            payroll[i] -
            rent[i] -
            utilities[i] -
            loan_emi[i] -
            other_expenses[i]
        )
        closing_balance[i] = closing_cash
        current_cash = closing_cash
        
    df = pd.DataFrame({
        'date': dates,
        'type': ['historical'] * history_days + ['expected'] * future_days,
        'opening_balance': opening_balance,
        'customer_payments_in': customer_payments_in,
        'supplier_payments': supplier_payments,
        'payroll': payroll,
        'rent': rent,
        'utilities': utilities,
        'loan_emi': loan_emi,
        'other_expenses': other_expenses,
        'closing_balance': closing_balance
    })
    
    # Round everything to 2 decimals
    cols_to_round = ['opening_balance', 'customer_payments_in', 'supplier_payments', 
                     'payroll', 'rent', 'utilities', 'loan_emi', 'other_expenses', 'closing_balance']
    df[cols_to_round] = df[cols_to_round].round(2)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "sample_msme_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to {output_path}")
    print(f"Total days generated: {total_days} ({history_days} historical, {future_days} expected)")
    
if __name__ == "__main__":
    generate_msme_data()
