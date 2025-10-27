import pandas as pd
import numpy as np
import random

print("Generating synthetic dataset...")

# Define project parameters
N_ROWS = 5000
PRODUCTS = ['Product A', 'Product B', 'Product C']
BASE_PRICES = {'Product A': 120, 'Product B': 80, 'Product C': 150}
ELASTICITIES = {'Product A': -1.8, 'Product B': -2.5, 'Product C': -1.2} # How sensitive demand is to price

data = []
start_date = pd.to_datetime('2023-01-01')

for i in range(N_ROWS):
    # --- Base Features ---
    date = start_date + pd.to_timedelta(i % 730, unit='D') # 2 years of data
    product_id = random.choice(PRODUCTS)
    day_of_week = date.dayofweek # 0=Monday, 6=Sunday
    
    # --- Causal Features (Our "levers" and market conditions) ---
    is_holiday = 1 if date.month == 12 or (date.month == 11 and date.day > 20) else 0
    is_promotion = 1 if (i % 10 == 0) or (day_of_week == 4) else 0 # Promotions on Fridays or every 10th day
    
    base_price = BASE_PRICES[product_id]
    
    # Simulate competitor price: normally distributed around our base price
    competitor_price = round(np.random.normal(base_price * 1.05, 5), 2)
    
    # Simulate our price: varies around the base price
    our_price = round(np.random.normal(base_price, 10), 2)
    
    # Simulate inventory
    inventory_level = np.random.randint(0, 1000)
    
    # --- Target Variable (units_sold) ---
    # This is the "secret sauce" of our simulation. Demand is a function of all features.
    
    # 1. Base demand
    base_demand = 150
    if product_id == 'Product B':
        base_demand = 250
    if product_id == 'Product C':
        base_demand = 100
        
    # 2. Price Elasticity Effect (Log-Log Model)
    # The ratio of our price to competitor's price is key
    price_ratio = our_price / competitor_price
    elasticity_effect = price_ratio ** ELASTICITIES[product_id]
    
    # 3. Other Effects
    promo_effect = 1.5 if is_promotion else 1.0
    holiday_effect = 1.8 if is_holiday else 1.0
    day_of_week_effect = 1.0
    if day_of_week in [5, 6]: # Weekend
        day_of_week_effect = 1.4
        
    # Combine all effects
    calculated_demand = base_demand * elasticity_effect * promo_effect * holiday_effect * day_of_week_effect
    
    # Add random noise
    noise = np.random.normal(1.0, 0.1)
    calculated_demand = calculated_demand * noise
    
    # Ensure demand is positive and an integer
    units_sold = int(np.floor(max(0, calculated_demand)))
    
    # Sales are capped by inventory
    units_sold = min(units_sold, inventory_level)
    
    # Add to our dataset
    data.append({
        'date': date,
        'product_id': product_id,
        'day_of_week': day_of_week,
        'is_holiday': is_holiday,
        'is_promotion': is_promotion,
        'inventory_level': inventory_level,
        'competitor_price': competitor_price,
        'our_price': our_price,
        'units_sold': units_sold
    })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('synthetic_ecommerce_data.csv', index=False)

print(f"Successfully generated and saved 'synthetic_ecommerce_data.csv' with {len(df)} rows.")
print(df.head())