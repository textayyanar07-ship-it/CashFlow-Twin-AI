import pandas as pd
import numpy as np
from datetime import timedelta

def simulate_cashflow(df, start_cash, supplier_pct_change=0.0, customer_delay_days=0, expense_pct_change=0.0, new_order_amount=0.0, new_order_day=0, emi_shift_days=0):
    """
    Simulates forward cashflow given a dataframe of future transactions and scenario adjustments.
    
    Args:
        df: DataFrame of expected transactions (must contain 'expected' type and be sorted by date)
        start_cash: Initial opening balance for the first day of simulation
        supplier_pct_change: Percentage change in supplier payments (e.g., -10 for a 10% reduction)
        customer_delay_days: Shift customer payments forward by this many days
        expense_pct_change: Percentage change in other expenses
        new_order_amount: A one-time lump sum customer payment
        new_order_day: Which day index (from start) the new order arrives
        emi_shift_days: Shift loan EMI payments forward by this many days
        
    Returns:
        sim_df: DataFrame with updated 'opening_balance' and 'closing_balance'
        runway_days: Number of days until cash goes negative (or len(df) if it doesn't)
        deficit_date: Date of first negative balance (or None)
        deficit_amount: Absolute amount of the first negative balance (or None)
    """
    df = df.copy()
    
    # Apply customer payment delay
    if customer_delay_days > 0:
        # Shift values forward. The last few days will be lost, first few days will be 0.
        shifted_payments = np.roll(df['customer_payments_in'].values, customer_delay_days)
        shifted_payments[:customer_delay_days] = 0
        df['customer_payments_in'] = shifted_payments

    # Apply supplier percentage change
    if supplier_pct_change != 0:
        factor = 1 + (supplier_pct_change / 100.0)
        df['supplier_payments'] = df['supplier_payments'] * factor
        
    # Apply expense percentage change
    if expense_pct_change != 0:
        factor = 1 + (expense_pct_change / 100.0)
        # Apply to non-fixed expenses like utilities and other_expenses
        df['utilities'] = df['utilities'] * factor
        df['other_expenses'] = df['other_expenses'] * factor
        
    # Apply EMI shift
    if emi_shift_days != 0:
        shifted_emi = np.roll(df['loan_emi'].values, emi_shift_days)
        if emi_shift_days > 0:
            shifted_emi[:emi_shift_days] = 0
        else:
            shifted_emi[emi_shift_days:] = 0
        df['loan_emi'] = shifted_emi
        
    # Apply new order
    if new_order_amount > 0 and 0 <= new_order_day < len(df):
        df.iloc[new_order_day, df.columns.get_loc('customer_payments_in')] += new_order_amount
        
    # Calculate daily cash
    runway_days = None
    deficit_date = None
    deficit_amount = None
    
    current_cash = start_cash
    start_date = df.iloc[0]['date']
    
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('opening_balance')] = current_cash
        
        inflow = df.iloc[i]['customer_payments_in']
        outflow = (
            df.iloc[i]['supplier_payments'] +
            df.iloc[i]['payroll'] +
            df.iloc[i]['rent'] +
            df.iloc[i]['utilities'] +
            df.iloc[i]['loan_emi'] +
            df.iloc[i]['other_expenses']
        )
        
        closing_cash = current_cash + inflow - outflow
        df.iloc[i, df.columns.get_loc('closing_balance')] = closing_cash
        
        if closing_cash < 0 and runway_days is None:
            # First negative balance
            deficit_date = df.iloc[i]['date']
            runway_days = (deficit_date - start_date).days
            deficit_amount = abs(closing_cash)
            
        current_cash = closing_cash
        
    if runway_days is None:
        runway_days = len(df)
        
    return df, runway_days, deficit_date, deficit_amount
