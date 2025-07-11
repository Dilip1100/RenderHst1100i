import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import os
import tempfile
from faker import Faker
from datetime import datetime
import random
import json

# Configure Plotly for offline rendering
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Set up logging
log_dir = os.path.join(tempfile.gettempdir(), "AutomotiveDashboard")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "dashboard.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Faker
fake = Faker()

# Define the main dashboard class
class AutomotiveDashboard:
    def __init__(self):
        self.df = self.generate_sales_data()
        if self.df.empty:
            st.error("Failed to generate sales data. Check logs.")
            logging.error("Failed to generate sales data")
            return
        self.hr_data, self.inventory_data, self.crm_data, self.demo_data, self.time_log_data = self.generate_fake_data()
        self.filtered_df = self.df.copy()

    def generate_sales_data(self):
        try:
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
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Year'] = df['Date'].dt.year
            df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            logging.info("Sales data generated successfully")
            return df
        except Exception as e:
            logging.error(f"Error generating sales data: {str(e)}")
            return pd.DataFrame()

    def generate_fake_data(self):
        try:
            hr_data = pd.DataFrame({
                "Employee ID": [f"E{1000+i}" for i in range(10)],
                "Name": [fake.name() for _ in range(10)],
                "Role": ["Sales Exec", "Manager", "Technician", "Clerk", "Sales Exec", "Technician", "HR", "Manager", "Clerk", "Sales Exec"],
                "Department": ["Sales", "Sales", "Service", "Admin", "Sales", "Service", "HR", "Sales", "Admin", "Sales"],
                "Join Date": pd.date_range(start="2018-01-01", periods=10, freq="180D"),
                "Salary (USD)": [50000 + i*1500 for i in range(10)],
                "Performance Score": [round(x, 1) for x in np.random.uniform(2.5, 5.0, 10)]
            })
            time_log_data = pd.DataFrame({
                "Employee ID": np.random.choice(hr_data["Employee ID"], size=30, replace=True),
                "Date": pd.date_range(end=pd.to_datetime("2025-07-07"), periods=30).tolist(),
                "Clock In": [f"{np.random.randint(8, 10)}:{str(np.random.randint(0, 60)).zfill(2)} AM" for _ in range(30)],
                "Clock Out": [f"{np.random.randint(4, 6)+12}:{str(np.random.randint(0, 60)).zfill(2)} PM" for _ in range(30)],
                "Total Hours": [round(np.random.uniform(6.5, 9.5), 1) for _ in range(30)]
            }).sort_values(by="Date", ascending=False)
            inventory_data = pd.DataFrame({
                "Part ID": [f"P{i:04d}" for i in range(1, 21)],
                "Part Name": [fake.word().capitalize() + " " + random.choice(["Filter", "Brake", "Tire", "Battery", "Sensor", "Pump"]) for _ in range(20)],
                "Car Make": [random.choice(self.df['Car Make'].dropna().unique()) for _ in range(20)],
                "Stock Level": [random.randint(0, 150) for _ in range(20)],
                "Reorder Level": [random.randint(10, 60) for _ in range(20)],
                "Unit Cost": [round(random.uniform(20, 600), 2) for _ in range(20)]
            })
            end_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
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
            logging.info("Fake data generated successfully")
            return hr_data, inventory_data, crm_data, demo_data, time_log_data
        except Exception as e:
            logging.error(f"Error generating fake data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def update_car_models(self, car_make):
        try:
            models = ['All']
            if car_make != 'All':
                models += sorted(self.df[self.df['Car Make'] == car_make]['Car Model'].dropna().unique())
            logging.info(f"Car models updated for {car_make}")
            return models
        except Exception as e:
            logging.error(f"Error updating car models: {str(e)}")
            st.error(f"Failed to update car models: {str(e)}")
            return ['All']

    def apply_filters(self, salesperson, car_make, car_model, car_year, metric):
        try:
            self.filtered_df = self.df.copy()
            if salesperson != 'All':
                self.filtered_df = self.filtered_df[self.filtered_df['Salesperson'] == salesperson]
            if car_make != 'All':
                self.filtered_df = self.filtered_df[self.filtered_df['Car Make'] == car_make]
            if car_model != 'All':
                self.filtered_df = self.filtered_df[self.filtered_df['Car Model'] == car_model]
            if car_year != 'All':
                self.filtered_df = self.filtered_df[self.filtered_df['Car Year'].astype(str) == car_year]
            logging.info("Filters applied successfully")
            return {
                'total_sales': f"${self.filtered_df['Sale Price'].sum():,.0f}",
                'total_comm': f"${self.filtered_df['Commission Earned'].sum():,.0f}",
                'avg_price': f"${self.filtered_df['Sale Price'].mean():,.0f}" if not self.filtered_df.empty else "$0",
                'trans_count': f"{self.filtered_df.shape[0]:,}"
            }
        except Exception as e:
            logging.error(f"Error applying filters: {str(e)}")
            st.error(f"Failed to apply filters: {str(e)}")
            return {'total_sales': '$0', 'total_comm': '$0', 'avg_price': '$0', 'trans_count': '0'}

    def render_charts_and_tables(self, metric):
        try:
            # KPI Trend
            with st.expander("KPI Trend", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for KPI Trend")
                else:
                    kpi_trend = self.filtered_df.groupby('Month')[['Sale Price', 'Commission Earned']].sum().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Sale Price'], name='Sale Price', line=dict(color='#A9A9A9')))
                    fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Commission Earned'], name='Commission', line=dict(color='#808080')))
                    fig.update_layout(
                        xaxis_title='Month', yaxis_title='Amount ($)', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("KPI trend chart rendered successfully")

            # 3D Sales
            with st.expander("3D Sales", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for 3D Sales")
                else:
                    scatter_data = self.filtered_df.sample(n=min(100, len(self.filtered_df)), random_state=1)
                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=scatter_data['Commission Earned'], y=scatter_data['Sale Price'], z=scatter_data['Car Year'],
                            mode='markers', marker=dict(size=5, color=scatter_data['Car Year'], colorscale='Greys', showscale=True)
                        )
                    ])
                    fig.update_layout(
                        scene=dict(xaxis_title='Commission Earned ($)', yaxis_title='Sale Price ($)', zaxis_title='Car Year'),
                        template='plotly_dark', plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("3D sales chart rendered successfully")

            # Heatmap
            with st.expander("Heatmap", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for Heatmap")
                else:
                    heatmap_data = self.filtered_df.pivot_table(
                        values=metric, index='Salesperson', columns='Car Make', aggfunc='sum', fill_value=0
                    )
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Greys'
                    ))
                    fig.update_layout(
                        xaxis_title='Car Make', yaxis_title='Salesperson', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Heatmap chart rendered successfully")

            # Top Performers
            with st.expander("Top Performers", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for Top Performers")
                else:
                    top_salespeople = self.filtered_df.groupby('Salesperson')[metric].sum().nlargest(10).reset_index()
                    fig = go.Figure(data=[go.Bar(x=top_salespeople['Salesperson'], y=top_salespeople[metric], marker_color='#A9A9A9')])
                    fig.update_layout(
                        xaxis_title='Salesperson', yaxis_title=f"{metric} ($)", template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Top performers chart rendered successfully")

            # Vehicle Sales
            with st.expander("Vehicle Sales", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for Vehicle Sales")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        car_make_metric = self.filtered_df.groupby('Car Make')['Sale Price'].sum().nlargest(10).reset_index()
                        fig = go.Figure(data=go.Pie(
                            labels=car_make_metric['Car Make'], values=car_make_metric['Sale Price'],
                            marker_colors=['#D3D3D3', '#A9A9A9', '#808080', '#606060', '#4A4A4A', '#3A3A3A', '#2A2A2A', '#1C1C1C']
                        ))
                        fig.update_layout(template='plotly_dark', plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        logging.info("Car make pie chart rendered successfully")
                    with col2:
                        car_model_metric = self.filtered_df.groupby('Car Model')['Sale Price'].sum().nlargest(10).reset_index()
                        fig = go.Figure(data=go.Pie(
                            labels=car_model_metric['Car Model'], values=car_model_metric['Sale Price'],
                            marker_colors=['#D3D3D3', '#A9A9A9', '#808080', '#606060', '#4A4A4A', '#3A3A3A', '#2A2A2A', '#1C1C1C']
                        ))
                        fig.update_layout(template='plotly_dark', plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        logging.info("Car model pie chart rendered successfully")

            # Model Comparison
            with st.expander("Model Comparison", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for Model Comparison")
                else:
                    model_comparison = self.filtered_df.groupby(['Car Make', 'Car Model']).agg({
                        'Sale Price': ['mean', 'sum', 'count'],
                        'Commission Earned': 'mean'
                    }).round(2)
                    model_comparison.columns = ['Avg Sale Price', 'Total Sales', 'Transaction Count', 'Avg Commission']
                    model_comparison = model_comparison.reset_index()
                    model_comparison['Avg Sale Price'] = model_comparison['Avg Sale Price'].apply(lambda x: f"${x:,.2f}")
                    model_comparison['Total Sales'] = model_comparison['Total Sales'].apply(lambda x: f"${x:,.2f}")
                    model_comparison['Avg Commission'] = model_comparison['Avg Commission'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(model_comparison, use_container_width=True)
                    logging.info("Model comparison table rendered successfully")

            # Trends
            with st.expander("Trends", expanded=False):
                if self.filtered_df.empty:
                    st.write("No data available for Trends")
                else:
                    st.subheader("Quarter-over-Quarter Trend")
                    trend_df = self.filtered_df.groupby('Quarter')[['Sale Price', 'Commission Earned']].sum().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=trend_df['Quarter'], y=trend_df['Sale Price'], name='Sale Price', line=dict(color='#A9A9A9')))
                    fig.add_trace(go.Scatter(x=trend_df['Quarter'], y=trend_df['Commission Earned'], name='Commission', line=dict(color='#808080')))
                    fig.update_layout(
                        xaxis_title='Quarter', yaxis_title='Amount ($)', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Trends chart rendered successfully")

                    st.subheader("Quarter-over-Quarter % Change")
                    trend_df['Sale Price QoQ %'] = trend_df['Sale Price'].pct_change().fillna(0) * 100
                    trend_df['Commission QoQ %'] = trend_df['Commission Earned'].pct_change().fillna(0) * 100
                    qoq_table = trend_df[['Quarter', 'Sale Price QoQ %', 'Commission QoQ %']].copy()
                    qoq_table['Sale Price QoQ %'] = qoq_table['Sale Price QoQ %'].apply(lambda x: f"{x:.2f}%")
                    qoq_table['Commission QoQ %'] = qoq_table['Commission QoQ %'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(qoq_table, use_container_width=True)
                    logging.info("QoQ table rendered successfully")

                    st.subheader("Monthly Trend")
                    monthly_trend = self.filtered_df.groupby('Month')[['Sale Price', 'Commission Earned']].sum().reset_index()
                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(go.Bar(x=monthly_trend['Month'], y=monthly_trend['Sale Price'], name='Sale Price', marker_color='#A9A9A9'))
                    fig.add_trace(go.Bar(x=monthly_trend['Month'], y=monthly_trend['Commission Earned'], name='Commission', marker_color='#808080'))
                    fig.update_layout(
                        xaxis_title='Month', yaxis_title='Amount ($)', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'),
                        barmode='group', height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Monthly trend chart rendered successfully")

            # HR Overview
            with st.expander("HR Overview", expanded=False):
                if self.hr_data.empty:
                    st.write("No data available for HR")
                else:
                    st.subheader("Employee Information & Salary")
                    hr_display = self.hr_data.copy()
                    hr_display['Salary (USD)'] = hr_display['Salary (USD)'].apply(lambda x: f"${x:,.2f}")
                    hr_display['Join Date'] = hr_display['Join Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(hr_display, use_container_width=True)
                    logging.info("HR table rendered successfully")

                    st.subheader("Performance Distribution")
                    fig = go.Figure(data=[go.Histogram(x=self.hr_data['Performance Score'], nbinsx=5, marker_color='#A9A9A9')])
                    fig.update_layout(
                        xaxis_title='Performance Score', yaxis_title='Count', template='plotly_dark',
                        plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("HR performance chart rendered successfully")

                    st.subheader("Employee Time Log")
                    time_log_display = self.time_log_data.copy()
                    time_log_display['Date'] = time_log_display['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(time_log_display, use_container_width=True)
                    logging.info("Time log table rendered successfully")

                    st.subheader("Total Logged Hours per Employee")
                    if self.time_log_data.empty:
                        st.write("No time log data available")
                    else:
                        total_hours = self.time_log_data.groupby('Employee ID')['Total Hours'].sum().reset_index()
                        fig = go.Figure(data=[go.Bar(x=total_hours['Employee ID'], y=total_hours['Total Hours'], marker_color='#A9A9A9')])
                        fig.update_layout(
                            xaxis_title='Employee ID', yaxis_title='Total Hours', template='plotly_dark',
                            xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        logging.info("HR hours chart rendered successfully")

            # Inventory
            with st.expander("Inventory", expanded=False):
                if self.inventory_data.empty:
                    st.write("No data available for Inventory")
                else:
                    inventory_display = self.inventory_data.copy()
                    inventory_display['Unit Cost'] = inventory_display['Unit Cost'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(inventory_display, use_container_width=True)
                    logging.info("Inventory table rendered successfully")

                    st.subheader("Low Stock Alert")
                    low_stock = self.inventory_data[self.inventory_data['Stock Level'] < self.inventory_data['Reorder Level']]
                    if low_stock.empty:
                        st.write("No low stock items")
                    else:
                        fig = go.Figure(data=[go.Bar(x=low_stock['Part Name'], y=low_stock['Stock Level'], marker_color='#A9A9A9')])
                        fig.update_layout(
                            xaxis_title='Part Name', yaxis_title='Stock Level', template='plotly_dark',
                            xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        logging.info("Low stock chart rendered successfully")

            # CRM
            with st.expander("CRM", expanded=False):
                if self.crm_data.empty:
                    st.write("No data available for CRM")
                else:
                    crm_display = self.crm_data.copy()
                    crm_display['Contact Date'] = crm_display['Contact Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                    st.dataframe(crm_display, use_container_width=True)
                    logging.info("CRM table rendered successfully")

                    st.subheader("Satisfaction Over Time")
                    line_chart_data = self.crm_data.copy()
                    line_chart_data['Contact Date'] = pd.to_datetime(line_chart_data['Contact Date'])
                    line_chart_data = line_chart_data.groupby('Contact Date')['Satisfaction Score'].mean().reset_index()
                    fig = go.Figure(data=[go.Scatter(x=line_chart_data['Contact Date'], y=line_chart_data['Satisfaction Score'], mode='lines+markers', line=dict(color='#A9A9A9'))])
                    fig.update_layout(
                        xaxis_title='Contact Date', yaxis_title='Satisfaction Score', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Satisfaction over time chart rendered successfully")

                    st.subheader("Satisfaction Score by Interaction Type")
                    interaction_types = self.crm_data['Interaction Type'].unique()
                    fig = go.Figure()
                    for itype in interaction_types:
                        fig.add_trace(go.Box(y=self.crm_data[self.crm_data['Interaction Type'] == itype]['Satisfaction Score'], name=itype))
                    fig.update_layout(
                        xaxis_title='Interaction Type', yaxis_title='Satisfaction Score', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Satisfaction by type chart rendered successfully")

            # Demographics
            with st.expander("Demographics", expanded=False):
                if self.demo_data.empty:
                    st.write("No data available for Demographics")
                else:
                    demo_display = self.demo_data.copy()
                    demo_display['Purchase Amount'] = demo_display['Purchase Amount'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(demo_display, use_container_width=True)
                    logging.info("Demographics table rendered successfully")

                    st.subheader("Age Group Distribution")
                    age_counts = self.demo_data['Age Group'].value_counts().reset_index()
                    age_counts.columns = ['Age Group', 'Count']
                    fig = go.Figure(data=[go.Bar(x=age_counts['Age Group'], y=age_counts['Count'], marker_color='#A9A9A9')])
                    fig.update_layout(
                        xaxis_title='Age Group', yaxis_title='Count', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Age distribution chart rendered successfully")

                    st.subheader("Purchase Amount by Region")
                    regions = self.demo_data['Region'].unique()
                    fig = go.Figure()
                    for region in regions:
                        fig.add_trace(go.Box(y=self.demo_data[self.demo_data['Region'] == region]['Purchase Amount'], name=region))
                    fig.update_layout(
                        xaxis_title='Region', yaxis_title='Purchase Amount ($)', template='plotly_dark',
                        xaxis=dict(tickangle=45), plot_bgcolor='#2A2A2A', paper_bgcolor='#2A2A2A', font=dict(color='#D3D3D3'), height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    logging.info("Purchase amount by region chart rendered successfully")

        except Exception as e:
            logging.error(f"Error rendering charts and tables: {str(e)}")
            st.error(f"Failed to render charts and tables: {str(e)}")

    def download_csv(self):
        try:
            csv = self.filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
            logging.info("CSV download button rendered")
        except Exception as e:
            logging.error(f"Error preparing CSV download: {str(e)}")
            st.error(f"Failed to prepare CSV download: {str(e)}")

def main():
    st.set_page_config(page_title="Automotive Analytics Dashboard", page_icon="🚗", layout="wide")
    st.markdown("""
        <style>
        .main {background-color: #1C1C1C; color: #D3D3D3; font-family: 'Segoe UI';}
        .stButton>button {
            background-color: #2A2A2A; color: #D3D3D3;
            border: 1px solid #4A4A4A; border-radius: 5px; padding: 5px;
        }
        .stButton>button:hover {background-color: #606060;}
        .stSelectbox [data-baseweb="select"] {background-color: #2A2A2A; color: #D3D3D3;}
        </style>
        """, unsafe_allow_html=True)

    st.title("🚗 Automotive Analytics Dashboard")
    dashboard = AutomotiveDashboard()

    # Filters
    st.header("🔍 Filter Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        salesperson = st.selectbox("Salesperson", ['All'] + sorted(dashboard.df['Salesperson'].dropna().unique()))
    with col2:
        car_make = st.selectbox("Car Make", ['All'] + sorted(dashboard.df['Car Make'].dropna().unique()))
    with col3:
        car_year = st.selectbox("Car Year", ['All'] + sorted(dashboard.df['Car Year'].dropna().astype(str).unique()))
    car_model_options = dashboard.update_car_models(car_make)
    metric = st.selectbox("Metric", ["Sale Price", "Commission Earned"])
    col4, col5 = st.columns([1, 2])
    with col4:
        car_model = st.selectbox("Car Model", car_model_options)
    with col5:
        if st.button("Apply Filters"):
            metrics = dashboard.apply_filters(salesperson, car_make, car_model, car_year, metric)
            st.session_state.metrics = metrics
    if 'metrics' in st.session_state:
        st.header("📌 Key Performance Indicators")
        col_metrics = st.columns(4)
        col_metrics[0].metric("Total Sales", st.session_state.metrics['total_sales'])
        col_metrics[1].metric("Total Commission", st.session_state.metrics['total_comm'])
        col_metrics[2].metric("Avg Sale Price", st.session_state.metrics['avg_price'])
        col_metrics[3].metric("Transactions", st.session_state.metrics['trans_count'])

    # Render charts and tables
    dashboard.render_charts_and_tables(metric)

    # Download button
    dashboard.download_csv()

    # Footer
    st.markdown("© 2025 One Trust | Crafted for smarter auto-financial decisions", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
