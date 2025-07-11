# app.py (Streamlit version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from faker import Faker
from datetime import datetime
import random

# Configure Plotly
pio.templates.default = "plotly_dark"

# Initialize Faker
fake = Faker()

class AutomotiveDashboard:
    def __init__(self):
        self.df = self.generate_sales_data()
        self.hr_data, self.inventory_data, self.crm_data, self.demo_data, self.time_log_data = self.generate_fake_data()

    def generate_sales_data(self):
        car_makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Hyundai', 'Volkswagen']
        car_models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4'],
            'Honda': ['Civic', 'Accord', 'CR-V'],
            'Ford': ['F-150', 'Mustang', 'Explorer'],
            'Chevrolet': ['Silverado', 'Malibu', 'Equinox'],
            'BMW': ['3 Series', '5 Series', 'X5'],
            'Mercedes': ['C-Class', 'E-Class', 'GLC'],
            'Hyundai': ['Elantra', 'Sonata', 'Tucson'],
            'Volkswagen': ['Jetta', 'Passat', 'Tiguan']
        }
        salespeople = [fake.name() for _ in range(10)]
        dates = pd.date_range(start="2023-01-01", end="2025-07-07", freq="D")
        data = {
            'Salesperson': [random.choice(salespeople) for _ in range(1000)],
            'Car Make': [random.choice(car_makes) for _ in range(1000)],
            'Car Year': [random.randint(2018, 2025) for _ in range(1000)],
            'Date': [random.choice(dates) for _ in range(1000)],
            'Sale Price': [round(random.uniform(15000, 100000), 2) for _ in range(1000)],
            'Commission Earned': [round(random.uniform(500, 5000), 2) for _ in range(1000)]
        }
        df = pd.DataFrame(data)
        df['Car Model'] = df['Car Make'].apply(lambda x: random.choice(car_models[x]))
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        return df

    def generate_fake_data(self):
        hr_data = pd.DataFrame({
            "Employee ID": [f"E{1000+i}" for i in range(10)],
            "Name": [fake.name() for _ in range(10)],
            "Role": ["Sales Exec", "Manager", "Technician", "Clerk", "Sales Exec", "Technician", "HR", "Manager", "Clerk", "Sales Exec"],
            "Department": ["Sales", "Sales", "Service", "Admin", "Sales", "Service", "HR", "Sales", "Admin", "Sales"],
            "Join Date": pd.date_range(start="2018-01-01", periods=10, freq="180D"),
            "Salary (USD)": [50000 + i*1500 for i in range(10)],
            "Performance Score": [round(x, 1) for x in np.random.uniform(2.5, 5.0, 10)]
        })

        end_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
        time_log_data = pd.DataFrame({
            "Employee ID": np.random.choice(hr_data["Employee ID"], size=30, replace=True),
            "Date": pd.date_range(end=end_date, periods=30),
            "Clock In": [f"{np.random.randint(8, 10)}:{str(np.random.randint(0, 60)).zfill(2)} AM" for _ in range(30)],
            "Clock Out": [f"{np.random.randint(4, 6)+12}:{str(np.random.randint(0, 60)).zfill(2)} PM" for _ in range(30)],
            "Total Hours": [round(np.random.uniform(6.5, 9.5), 1) for _ in range(30)]
        })

        inventory_data = pd.DataFrame({
            "Part ID": [f"P{i:04d}" for i in range(1, 21)],
            "Part Name": [fake.word().capitalize() + " " + random.choice(["Filter", "Brake", "Tire", "Battery", "Sensor", "Pump"]) for _ in range(20)],
            "Car Make": [random.choice(self.df['Car Make'].dropna().unique()) for _ in range(20)],
            "Stock Level": [random.randint(0, 150) for _ in range(20)],
            "Reorder Level": [random.randint(10, 60) for _ in range(20)],
            "Unit Cost": [round(random.uniform(20, 600), 2) for _ in range(20)]
        })

        crm_data = pd.DataFrame({
            "Customer ID": [f"C{100+i}" for i in range(20)],
            "Customer Name": [fake.name() for _ in range(20)],
            "Contact Date": [fake.date_between(start_date="-1y", end_date=end_date) for _ in range(20)],
            "Interaction Type": [random.choice(["Inquiry", "Complaint", "Follow-up", "Feedback", "Service Request"]) for _ in range(20)],
            "Salesperson": [random.choice(self.df['Salesperson'].dropna().unique()) for _ in range(20)],
            "Satisfaction Score": [round(random.uniform(1.0, 5.0), 1) for _ in range(20)]
        })

        demo_data = pd.DataFrame({
            "Customer ID": [f"C{100+i}" for i in range(20)],
            "Age Group": [random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]) for _ in range(20)],
            "Region": [fake.state() for _ in range(20)],
            "Purchase Amount": [round(random.uniform(15000, 100000), 2) for _ in range(20)],
            "Preferred Make": [random.choice(self.df['Car Make'].dropna().unique()) for _ in range(20)]
        })

        return hr_data, inventory_data, crm_data, demo_data, time_log_data


# Streamlit app starts here
st.set_page_config(page_title="Automotive Dashboard", layout="wide")
st.title("🚗 Automotive KPI Dashboard")

dashboard = AutomotiveDashboard()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
salesperson = st.sidebar.selectbox("Salesperson", ["All"] + sorted(dashboard.df['Salesperson'].unique()))
car_make = st.sidebar.selectbox("Car Make", ["All"] + sorted(dashboard.df['Car Make'].unique()))
car_model = st.sidebar.selectbox("Car Model", ["All"] + sorted(dashboard.df['Car Model'].unique()))
car_year = st.sidebar.selectbox("Car Year", ["All"] + sorted(dashboard.df['Car Year'].astype(str).unique()))

# Filter DataFrame
filtered_df = dashboard.df.copy()
if salesperson != "All":
    filtered_df = filtered_df[filtered_df["Salesperson"] == salesperson]
if car_make != "All":
    filtered_df = filtered_df[filtered_df["Car Make"] == car_make]
if car_model != "All":
    filtered_df = filtered_df[filtered_df["Car Model"] == car_model]
if car_year != "All":
    filtered_df = filtered_df[filtered_df["Car Year"].astype(str) == car_year]

# KPI summary
st.subheader("📊 KPI Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${filtered_df['Sale Price'].sum():,.0f}")
col2.metric("Total Commission", f"${filtered_df['Commission Earned'].sum():,.0f}")
col3.metric("Average Price", f"${filtered_df['Sale Price'].mean():,.0f}" if not filtered_df.empty else "$0")
col4.metric("Transactions", f"{filtered_df.shape[0]:,}")

# KPI Trend
st.subheader("📈 KPI Trend Over Time")
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    kpi_trend = filtered_df.groupby("Month")[["Sale Price", "Commission Earned"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Sale Price'], name="Sale Price", line=dict(color="#A9A9A9")))
    fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Commission Earned'], name="Commission", line=dict(color="#808080")))
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Amount ($)",
        xaxis=dict(tickangle=45), template="plotly_dark", height=450,
        plot_bgcolor="#2A2A2A", paper_bgcolor="#2A2A2A", font=dict(color="#D3D3D3")
    )
    st.plotly_chart(fig, use_container_width=True)

# Export CSV
st.download_button("📥 Download Filtered CSV", data=filtered_df.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv")
