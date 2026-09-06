def get_rule_based_recommendations(context):
    """
    Returns exactly 3 recommended actions based on the context (KPIs, deficit).
    """
    recs = []
    
    runway = context.get('runway_days', 999)
    if runway < 30:
        recs.append({
            'priority': '🔴',
            'problem': f'Critical cash shortage expected in {runway} days',
            'action': 'Delay non-essential supplier payments by 15 days',
            'impact': '+15 days of runway'
        })
    elif runway < 60:
        recs.append({
            'priority': '🟡',
            'problem': 'Cash runway is getting tight',
            'action': 'Review and cut discretionary expenses by 10%',
            'impact': '+10 days of runway'
        })
    else:
        recs.append({
            'priority': '🟢',
            'problem': 'Cash runway is stable',
            'action': 'Invest excess cash into short-term liquid funds',
            'impact': 'Yields ~5% annualized return'
        })
        
    recs.append({
        'priority': '🟡',
        'problem': 'Customer payments are contributing to cash drag',
        'action': 'Offer 2% discount for early payment on next 3 large invoices',
        'impact': 'Accelerates ₹50,000 by 15 days'
    })
    
    recs.append({
        'priority': '🟢',
        'problem': 'Upcoming recurring loan EMIs',
        'action': 'Evaluate refinancing options with current lower interest rates',
        'impact': 'Reduces monthly outflow by ₹2,000'
    })
    
    return recs[:3]
