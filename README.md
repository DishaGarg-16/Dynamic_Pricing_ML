# 📈 Dynamic Pricing Simulator for E-Commerce

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dynamicpricingml-5ypejqyqlfetgrdiv2uy5g.streamlit.app/)

This project is an interactive web application that uses a machine learning model to find the optimal price for e-commerce products.

It serves as a decision-support tool for a category manager. Instead of relying on guesswork, a manager can input current market conditions (like competitor prices, inventory levels, and day of the week) and the app will recommend a price that maximizes revenue.

## 🚀 Live Demo & Screenshot

**You can access the live application here:**
**[https://dynamicpricingml-5ypejqyqlfetgrdiv2uy5g.streamlit.app/](https://dynamicpricingml-5ypejqyqlfetgrdiv2uy5g.streamlit.app/)**

*(**To add a screenshot:** Take a picture of your app, save it as `screenshot.png` in this repository, then uncomment the line below by removing the ``)*

---

## Core Features

* **🤖 What-If Analysis:** Interactively change market factors like competitor pricing, inventory, and promotions to see the instant impact on the optimal price.
* **💸 Price Optimization:** Automatically scans a range of potential prices to find the single price point that maximizes estimated revenue.
* **📊 Data-Driven Demand Model:** Powered by an **XGBoost** regression model trained on historical sales data to predict customer demand at different price points.
* **📈 Interactive Visuals:** Generates "Price vs. Revenue" and "Price vs. Demand" charts to help users understand the price elasticity and the model's recommendation.

---

## Methodology

The project works in two main phases:

1.  **Demand Prediction (Offline Training):**
    * A synthetic dataset of e-commerce transactions is generated (`1_generate_data.py`).
    * An XGBoost model is trained on this data (`2_train_model.py`) to predict `units_sold` based on features like `our_price`, `competitor_price`, `day_of_week`, `product_id`, etc.
    * This trained model (a `*.pkl` file) is saved.

2.  **Revenue Optimization (Live in Streamlit):**
    * The Streamlit app (`app.py`) loads the pre-trained model.
    * The user provides the *current* market conditions (competitor price, etc.) in the sidebar.
    * The app runs a simulation: it programmatically tests a range of prices (e.g., $80 to $200).
    * For each price, it asks the model: "Given these conditions, how many units would we sell at *this price*?"
    * It calculates `estimated_revenue = price * predicted_demand` for all prices and recommends the price that results in the highest revenue.

---

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web app.
* **XGBoost:** For the machine learning (demand prediction) model.
* **Scikit-learn:** For the data preprocessing pipeline.
* **Pandas:** For data manipulation.

---

## 📂 Project Structure
├── 1_generate_data.py # Script to create synthetic_ecommerce_data.csv 
├── 2_train_model.py # Script to train the ML model and save the pipeline 
├── app.py # The main Streamlit application 
├── demand_model_pipeline.pkl # The final, trained model file (CRITICAL) 
├── requirements.txt # Python dependencies for deployment 
├── .gitignore # Files to be ignored by Git 
└── README.md # This file

---

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/dynamic-pricing-streamlit.git](https://github.com/YOUR_USERNAME/dynamic-pricing-streamlit.git)
    cd dynamic-pricing-streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(One-Time Setup)** You must generate the data and train the model locally first:
    ```bash
    python 1_generate_data.py
    python 2_train_model.py
    ```
    *(This creates the `demand_model_pipeline.pkl` file the app needs)*

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 📄 License

This project is licensed under the MIT License.
