import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Dynamic Pricing Simulator",
    page_icon="📈",
    layout="wide"
)

# --- Load Model ---
# Load the entire pipeline (preprocessor + model)
try:
    model_pipeline = joblib.load('demand_model_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file ('demand_model_pipeline.pkl') not found.")
    st.error("Please run 'python 2_train_model.py' to train and save the model first.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# --- App Title ---
st.title("📈 E-Commerce Dynamic Pricing Simulator")
st.markdown("Use this tool to find the optimal price for a product based on current market conditions.")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Step 1: Set Market Conditions")

product_id = st.sidebar.selectbox(
    'Select Product:',
    ('Product A', 'Product B', 'Product C')
)

day_of_week = st.sidebar.selectbox(
    'Day of the Week (0=Mon, 6=Sun):',
    (0, 1, 2, 3, 4, 5, 6)
)

competitor_price = st.sidebar.slider(
    'Average Competitor Price ($):',
    min_value=50.0, max_value=250.0, value=120.0, step=1.0
)

inventory_level = st.sidebar.slider(
    'Current Inventory Level:',
    min_value=0, max_value=1500, value=500
)

is_holiday = st.sidebar.checkbox('Is it a Holiday?')
is_promotion = st.sidebar.checkbox('Is a Promotion Active?')

# --- Optimization Parameters ---
st.sidebar.header("Step 2: Set Optimization")
st.sidebar.markdown("Define the price range to test.")

min_price = st.sidebar.number_input("Minimum Price to Test ($):", value=80.0)
max_price = st.sidebar.number_input("Maximum Price to Test ($):", value=200.0)
price_steps = st.sidebar.slider("Number of Prices to Test:", min_value=10, max_value=100, value=50)

if min_price >= max_price:
    st.sidebar.error("Max Price must be greater than Min Price.")
    st.stop()

# --- Run Simulation ---
if st.sidebar.button("Run Price Optimization", type="primary"):

    # 1. Create a DataFrame for simulation
    # We test every price in the defined range
    price_range = np.linspace(min_price, max_price, price_steps)
    
    # Create a DataFrame with all possible inputs
    # All columns must match the training data
    sim_data = {
        'product_id': product_id,
        'day_of_week': day_of_week,
        'is_holiday': 1 if is_holiday else 0,
        'is_promotion': 1 if is_promotion else 0,
        'inventory_level': inventory_level,
        'competitor_price': competitor_price,
        'our_price': price_range # This is the only column that changes
    }
    
    sim_df = pd.DataFrame(sim_data)

    # 2. Use the pipeline to predict demand
    # The pipeline handles all preprocessing internally
    try:
        predicted_demand = model_pipeline.predict(sim_df)
        
        # Ensure demand is not negative and is an integer
        sim_df['predicted_demand'] = np.floor(np.maximum(0, predicted_demand))
        
        # 3. Calculate predicted revenue
        sim_df['predicted_revenue'] = sim_df['our_price'] * sim_df['predicted_demand']

        # 4. Find the optimal price
        best_idx = sim_df['predicted_revenue'].idxmax()
        best_row = sim_df.loc[best_idx]
        
        best_price = best_row['our_price']
        best_revenue = best_row['predicted_revenue']
        best_demand = best_row['predicted_demand']

        # --- Display Results ---
        st.header(f"Optimal Price Recommendation for: {product_id}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Optimal Price 💰",
            value=f"${best_price:.2f}"
        )
        col2.metric(
            label="Estimated Revenue 💸",
            value=f"${best_revenue:,.2f}"
        )
        col3.metric(
            label="Predicted Units Sold 📦",
            value=f"{best_demand:,.0f} units"
        )
        
        st.markdown("---")

        # --- Charts ---
        st.subheader("Price vs. Predicted Revenue")
        st.line_chart(sim_df.rename(columns={'our_price': 'Price', 'predicted_revenue': 'Revenue'}).set_index('Price')['Revenue'])
        
        st.subheader("Price vs. Predicted Demand (Price Elasticity)")
        st.line_chart(sim_df.rename(columns={'our_price': 'Price', 'predicted_demand': 'Demand'}).set_index('Price')['Demand'])
        
        st.subheader("Simulation Data")
        st.dataframe(sim_df[['our_price', 'predicted_demand', 'predicted_revenue']].style.format({
            'our_price': '${:,.2f}',
            'predicted_demand': '{:,.0f}',
            'predicted_revenue': '${:,.2f}'
        }))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Set your parameters in the sidebar and click 'Run Price Optimization'.")