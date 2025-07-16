import sys
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
import logging
import os
import tempfile

# Configure Plotly for dark theme
pio.templates.default = "plotly_dark"

# Set up logging
log_dir = os.path.join(tempfile.gettempdir(), "SteelDashboard")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "dashboard.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Faker
fake = Faker('en_IN')

# Custom CSS for Midnight Sapphire Theme approximation in Streamlit
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #001F3F;
        color: #E0E0E0;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    /* Widgets */
    .stSelectbox, .stTextInput, .stButton button {
        background-color: #0A2F5A;
        color: #E0E0E0;
        border: 1px solid #A9A9A9;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #007BFF;
    }
    /* Tables */
    .dataframe {
        background-color: #001F3F;
        color: #E0E0E0;
    }
    /* Metrics */
    .stMetric {
        color: #E0E0E0;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #0A2F5A;
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #007BFF;
    }
    /* Expander */
    .stExpander {
        background-color: #001F3F;
        color: #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Steel Industry Analytics Dashboard", layout="wide")

def generate_sales_data():
    try:
        steel_types = ['Stainless Steel', 'Carbon Steel', 'Alloy Steel', 'Tool Steel', 'Galvanized Steel', 'Mild Steel', 'High-Strength Steel', 'Weathering Steel']
        steel_grades = {
            'Stainless Steel': ['304', '316', '430'],
            'Carbon Steel': ['A36', 'A572', 'A516'],
            'Alloy Steel': ['4140', '4340', '8620'],
            'Tool Steel': ['D2', 'O1', 'A2'],
            'Galvanized Steel': ['G90', 'G60', 'G30'],
            'Mild Steel': ['1018', '1020', '1045'],
            'High-Strength Steel': ['AH36', 'DH36', 'EH36'],
            'Weathering Steel': ['Corten A', 'Corten B', 'A588']
        }
        sales_reps = [fake.name() for _ in range(10)]
        dates = pd.date_range(start="2023-01-01", end="2025-07-07", freq="D")
        data = {
            'Sales Representative': [random.choice(sales_reps) for _ in range(1000)],
            'Steel Type': [random.choice(steel_types) for _ in range(1000)],
            'Production Year': [random.randint(2018, 2025) for _ in range(1000)],
            'Date': [random.choice(dates) for _ in range(1000)],
            'Sale Price': [round(random.uniform(100000, 10000000), 2) for _ in range(1000)],
            'Commission Earned': [round(random.uniform(5000, 50000), 2) for _ in range(1000)],
            'Customer Name': [fake.name() for _ in range(1000)]
        }
        df = pd.DataFrame(data)
        df['Steel Grade'] = df['Steel Type'].apply(lambda x: random.choice(steel_grades[x]))
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        logging.info("Sales data generated successfully")
        return df
    except Exception as e:
        logging.error(f"Error generating sales data: {str(e)}")
        return pd.DataFrame()

def generate_fake_data(df):
    try:
        tn_cities = [
            'Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tirunelveli',
            'Erode', 'Vellore', 'Thoothukudi', 'Dindigul', 'Thanjavur', 'Karur', 'Namakkal',
            'Tiruppur', 'Kanyakumari'
        ]
        hr_data = pd.DataFrame({
            "Employee ID": [f"E{1000+i}" for i in range(10)],
            "Name": [fake.name() for _ in range(10)],
            "Role": ["Sales Exec", "Manager", "Technician", "Clerk", "Sales Exec", "Technician", "HR", "Manager", "Clerk", "Sales Exec"],
            "Department": ["Sales", "Sales", "Service", "Admin", "Sales", "Service", "HR", "Sales", "Admin", "Sales"],
            "Join Date": pd.date_range(start="2018-01-01", periods=10, freq="180D"),
            "Salary (INR)": [300000 + i*15000 for i in range(10)],
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
            "Item ID": [f"I{i:04d}" for i in range(1, 21)],
            "Item Name": [fake.word().capitalize() + " " + random.choice(["Sheet", "Pipe", "Bar", "Coil", "Plate", "Rod"]) for _ in range(20)],
            "Steel Type": [random.choice(df['Steel Type'].dropna().unique()) for _ in range(20)],
            "Stock Level": [random.randint(0, 150) for _ in range(20)],
            "Reorder Level": [random.randint(10, 60) for _ in range(20)],
            "Unit Cost": [round(random.uniform(1500, 50000), 2) for _ in range(20)]
        })
        end_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
        crm_data = pd.DataFrame({
            "Customer ID": [f"C{100+i}" for i in range(20)],
            "Customer Name": [fake.name() for _ in range(20)],
            "Contact Date": [fake.date_between(start_date="-1y", end_date=end_date) for _ in range(20)],
            "Interaction Type": [random.choice(["Inquiry", "Complaint", "Follow-up", "Feedback", "Service Request"]) for _ in range(20)],
            "Sales Representative": [random.choice(df['Sales Representative'].dropna().unique()) for _ in range(20)],
            "Satisfaction Score": [round(random.uniform(1.0, 5.0), 1) for _ in range(20)]
        })
        demo_data = pd.DataFrame({
            "Customer ID": [f"C{100+i}" for i in range(20)],
            "Age Group": [random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]) for _ in range(20)],
            "Region": [random.choice(tn_cities) for _ in range(20)],
            "Purchase Amount": [round(random.uniform(100000, 10000000), 2) for _ in range(20)],
            "Preferred Type": [random.choice(df['Steel Type'].dropna().unique()) for _ in range(20)]
        })
        logging.info("Fake data generated successfully")
        return hr_data, time_log_data, inventory_data, crm_data, demo_data
    except Exception as e:
        logging.error(f"Error generating fake data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data with caching
@st.cache_data
def load_data():
    df = generate_sales_data()
    hr_data, time_log_data, inventory_data, crm_data, demo_data = generate_fake_data(df)
    return df, hr_data, time_log_data, inventory_data, crm_data, demo_data

df, hr_data, time_log_data, inventory_data, crm_data, demo_data = load_data()
total_overall_sales = df['Sale Price'].sum()

# Header
st.title("🔥 Steel Industry Analytics Dashboard")

# Help label
st.write("Welcome! Use the filters below to explore sales, performance, and more. Hover over elements for tips.")

# Filters
with st.expander("🔍 Filter Options", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        salespeople = ['All'] + sorted(df['Sales Representative'].dropna().unique().tolist())
        selected_salesperson = st.selectbox("Sales Representative", salespeople, help="Select a sales representative to filter data by their performance.")
        
        sale_years = ['All'] + sorted(df['Year'].dropna().astype(str).unique().tolist())
        selected_sale_year = st.selectbox("Sale Year", sale_years, help="Filter sales by the year they occurred.")
        
        production_years = ['All'] + sorted(df['Production Year'].dropna().astype(str).unique().tolist())
        selected_production_year = st.selectbox("Production Year", production_years, help="Filter by the year the steel was produced.")
        
        metrics = ["Sale Price", "Commission Earned"]
        selected_metric = st.selectbox("Metric", metrics, help="Choose the metric to analyze (e.g., total sales or commissions).")
    
    with col2:
        steel_types = ['All'] + sorted(df['Steel Type'].dropna().unique().tolist())
        selected_steel_type = st.selectbox("Steel Type", steel_types, help="Filter by steel type, such as Stainless or Carbon.")
        
        sale_months = ['All'] + sorted(df['Month'].dropna().unique().tolist())
        selected_sale_month = st.selectbox("Sale Month", sale_months, help="Filter sales by the month they occurred (format: YYYY-MM).")
        
        if selected_steel_type != 'All':
            grades = ['All'] + sorted(df[df['Steel Type'] == selected_steel_type]['Steel Grade'].dropna().unique().tolist())
        else:
            grades = ['All']
        selected_grade = st.selectbox("Steel Grade", grades, help="Filter by specific steel grade or variant.")
        
        search_query = st.text_input("Universal Search (e.g., 'Stainless', 'Chennai', '2024')", help="Search across all text fields for quick filtering.")

# Apply filters (Streamlit is reactive, so no apply button needed)
filtered_df = df.copy()
if selected_salesperson != 'All':
    filtered_df = filtered_df[filtered_df['Sales Representative'] == selected_salesperson]
if selected_steel_type != 'All':
    filtered_df = filtered_df[filtered_df['Steel Type'] == selected_steel_type]
if selected_grade != 'All':
    filtered_df = filtered_df[filtered_df['Steel Grade'] == selected_grade]
if selected_production_year != 'All':
    filtered_df = filtered_df[filtered_df['Production Year'].astype(str) == selected_production_year]
if selected_sale_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'].astype(str) == selected_sale_year]
if selected_sale_month != 'All':
    filtered_df = filtered_df[filtered_df['Month'] == selected_sale_month]
if search_query:
    string_cols = filtered_df.select_dtypes(include='object').columns
    mask = filtered_df.apply(lambda row: any(search_query.lower() in str(value).lower() for value in row[string_cols]), axis=1)
    filtered_df = filtered_df[mask]

# Metrics
total_sales = filtered_df['Sale Price'].sum()
total_comm = filtered_df['Commission Earned'].sum()
avg_price = filtered_df['Sale Price'].mean() if not filtered_df.empty else 0
trans_count = filtered_df.shape[0]
sales_percentage = (total_sales / total_overall_sales * 100) if total_overall_sales > 0 else 0

st.subheader("📌 Key Performance Indicators")
metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
with metrics_col1:
    st.metric("Total Sales", f"₹{total_sales:,.0f}")
with metrics_col2:
    st.metric("Total Commission", f"₹{total_comm:,.0f}")
with metrics_col3:
    st.metric("Avg Sale Price", f"₹{avg_price:,.0f}")
with metrics_col4:
    st.metric("Transactions", f"{trans_count:,}")
with metrics_col5:
    st.metric("Sales %", f"{sales_percentage:.2f}%")

# Tabs
tab_kpi, tab_3d, tab_heatmap, tab_top, tab_product, tab_grade, tab_trends, tab_hr, tab_inventory, tab_crm, tab_demo = st.tabs([
    "KPI Trend", "3D Sales", "Heatmap", "Top Performers", "Product Sales", "Grade Comparison", "Trends", 
    "HR Overview", "Inventory", "CRM", "Demographics"
])

# Common dark theme settings for Plotly
dark_theme_settings = {
    'template': 'plotly_dark',
    'plot_bgcolor': '#001F3F',
    'paper_bgcolor': '#001F3F',
    'font': dict(color='#E0E0E0'),
    'height': 600
}

with tab_kpi:
    if filtered_df.empty:
        st.write("No data available for KPI Trend")
    else:
        kpi_trend = filtered_df.groupby('Month')[['Sale Price', 'Commission Earned']].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Sale Price'], name='Sale Price', line=dict(color='#0078D7')))
        fig.add_trace(go.Scatter(x=kpi_trend['Month'], y=kpi_trend['Commission Earned'], name='Commission', line=dict(color='#E85D00')))
        fig.update_layout(xaxis_title='Month', yaxis_title='Amount (₹)', **dark_theme_settings, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

with tab_3d:
    if filtered_df.empty:
        st.write("No data available for 3D Sales")
    else:
        scatter_data = filtered_df.sample(n=min(100, len(filtered_df)), random_state=1)
        fig = go.Figure(data=[go.Scatter3d(
            x=scatter_data['Commission Earned'], y=scatter_data['Sale Price'], z=scatter_data['Production Year'],
            mode='markers', marker=dict(size=5, color=scatter_data['Production Year'], colorscale='Plasma', showscale=True)
        )])
        fig.update_layout(scene=dict(xaxis_title='Commission Earned (₹)', yaxis_title='Sale Price (₹)', zaxis_title='Production Year'), **dark_theme_settings)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Sales Transactions")
    sales_table_data = filtered_df[['Customer Name', 'Date', 'Sales Representative', 'Sale Price']].copy()
    sales_table_data['Date'] = sales_table_data['Date'].dt.strftime('%Y-%m-%d')
    sales_table_data['Sale Price'] = sales_table_data['Sale Price'].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(sales_table_data, use_container_width=True)

with tab_heatmap:
    if filtered_df.empty:
        st.write("No data available for Heatmap")
    else:
        heatmap_data = filtered_df.pivot_table(values=selected_metric, index='Sales Representative', columns='Steel Type', aggfunc='sum', fill_value=0)
        fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
        fig.update_layout(xaxis_title='Steel Type', yaxis_title='Sales Representative', **dark_theme_settings, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

with tab_top:
    if filtered_df.empty:
        st.write("No data available for Top Performers")
    else:
        top_sales_reps = filtered_df.groupby('Sales Representative')[selected_metric].sum().nlargest(10).reset_index()
        fig = go.Figure(data=[go.Bar(x=top_sales_reps['Sales Representative'], y=top_sales_reps[selected_metric], marker_color='#0078D7')])
        fig.update_layout(xaxis_title='Sales Representative', yaxis_title=f"{selected_metric} (₹)", **dark_theme_settings, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

with tab_product:
    if filtered_df.empty:
        st.write("No data available for Product Sales")
    else:
        col1, col2 = st.columns(2)
        with col1:
            steel_type_metric = filtered_df.groupby('Steel Type')['Sale Price'].sum().nlargest(10).reset_index()
            fig_type = go.Figure(data=go.Pie(labels=steel_type_metric['Steel Type'], values=steel_type_metric['Sale Price'], pull=[0.05]*len(steel_type_metric), marker_colors=px.colors.qualitative.Plotly))
            fig_type.update_layout(**dark_theme_settings)
            st.plotly_chart(fig_type, use_container_width=True)
        with col2:
            steel_grade_metric = filtered_df.groupby('Steel Grade')['Sale Price'].sum().nlargest(10).reset_index()
            fig_grade = go.Figure(data=go.Pie(labels=steel_grade_metric['Steel Grade'], values=steel_grade_metric['Sale Price'], pull=[0.05]*len(steel_grade_metric), marker_colors=px.colors.qualitative.Plotly))
            fig_grade.update_layout(**dark_theme_settings)
            st.plotly_chart(fig_grade, use_container_width=True)

with tab_grade:
    if filtered_df.empty:
        st.write("No data available for Grade Comparison")
    else:
        grade_comparison = filtered_df.groupby(['Steel Type', 'Steel Grade']).agg({
            'Sale Price': ['mean', 'sum', 'count'],
            'Commission Earned': 'mean'
        }).round(2)
        grade_comparison.columns = ['Avg Sale Price', 'Total Sales', 'Transaction Count', 'Avg Commission']
        grade_comparison = grade_comparison.reset_index()
        grade_comparison['Avg Sale Price'] = grade_comparison['Avg Sale Price'].apply(lambda x: f"₹{x:,.0f}")
        grade_comparison['Total Sales'] = grade_comparison['Total Sales'].apply(lambda x: f"₹{x:,.0f}")
        grade_comparison['Avg Commission'] = grade_comparison['Avg Commission'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(grade_comparison, use_container_width=True)

with tab_trends:
    if filtered_df.empty:
        st.write("No data available for Trends")
    else:
        st.subheader("Quarter-over-Quarter Trend")
        trend_df = filtered_df.groupby('Quarter')[['Sale Price', 'Commission Earned']].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_df['Quarter'], y=trend_df['Sale Price'], name='Sale Price', line=dict(color='#0078D7')))
        fig.add_trace(go.Scatter(x=trend_df['Quarter'], y=trend_df['Commission Earned'], name='Commission', line=dict(color='#E85D00')))
        fig.update_layout(xaxis_title='Quarter', yaxis_title='Amount (₹)', **dark_theme_settings, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Quarter-over-Quarter % Change")
        qoq_sales = filtered_df.groupby('Quarter')['Sale Price'].sum().reset_index()
        qoq_sales['QoQ Change %'] = qoq_sales['Sale Price'].pct_change() * 100
        qoq_sales = qoq_sales.iloc[1:].round(2)
        qoq_sales['Sale Price'] = qoq_sales['Sale Price'].apply(lambda x: f"₹{x:,.0f}")
        qoq_sales['QoQ Change %'] = qoq_sales['QoQ Change %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(qoq_sales, use_container_width=True)
        
        st.subheader("Monthly Trend")
        monthly_trend = filtered_df.groupby('Month')[['Sale Price', 'Commission Earned']].sum().reset_index()
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=monthly_trend['Month'], y=monthly_trend['Sale Price'], name='Sale Price', marker_color='#0078D7'))
        fig.add_trace(go.Bar(x=monthly_trend['Month'], y=monthly_trend['Commission Earned'], name='Commission', marker_color='#E85D00'))
        fig.update_layout(xaxis_title='Month', yaxis_title='Amount (₹)', **dark_theme_settings, xaxis=dict(tickangle=45), barmode='group')
        st.plotly_chart(fig, use_container_width=True)

with tab_hr:
    st.subheader("Employee Information & Salary")
    hr_display = hr_data.copy()
    hr_display['Join Date'] = hr_display['Join Date'].dt.strftime('%Y-%m-%d')
    hr_display['Salary (INR)'] = hr_display['Salary (INR)'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(hr_display, use_container_width=True)
    
    st.subheader("Performance Distribution")
    fig = go.Figure(data=[go.Histogram(x=hr_data['Performance Score'], nbinsx=5, marker_color='#0078D7')])
    fig.update_layout(xaxis_title='Performance Score', yaxis_title='Count', **dark_theme_settings)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Employee Time Log")
    time_log_display = time_log_data.copy()
    time_log_display['Date'] = time_log_display['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(time_log_display, use_container_width=True)
    
    st.subheader("Total Logged Hours per Employee")
    total_hours = time_log_data.groupby('Employee ID')['Total Hours'].sum().reset_index()
    fig = go.Figure(data=[go.Bar(x=total_hours['Employee ID'], y=total_hours['Total Hours'], marker_color='#0078D7')])
    fig.update_layout(xaxis_title='Employee ID', yaxis_title='Total Hours', **dark_theme_settings, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

with tab_inventory:
    inventory_display = inventory_data.copy()
    inventory_display['Unit Cost'] = inventory_display['Unit Cost'].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(inventory_display, use_container_width=True)
    
    st.subheader("Low Stock Alert")
    low_stock = inventory_data[inventory_data['Stock Level'] < inventory_data['Reorder Level']]
    if low_stock.empty:
        st.write("No low stock items")
    else:
        fig = go.Figure(data=[go.Bar(x=low_stock['Item Name'], y=low_stock['Stock Level'], marker_color='#E85D00')])
        fig.update_layout(xaxis_title='Item Name', yaxis_title='Stock Level', **dark_theme_settings, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

with tab_crm:
    crm_display = crm_data.copy()
    crm_display['Contact Date'] = crm_display['Contact Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)
    st.dataframe(crm_display, use_container_width=True)
    
    st.subheader("Satisfaction Over Time")
    line_chart_data = crm_data.copy()
    line_chart_data['Contact Date'] = pd.to_datetime(line_chart_data['Contact Date'])
    line_chart_data = line_chart_data.groupby('Contact Date')['Satisfaction Score'].mean().reset_index()
    fig = go.Figure(data=[go.Scatter(x=line_chart_data['Contact Date'], y=line_chart_data['Satisfaction Score'], mode='lines+markers', line=dict(color='#0078D7'))])
    fig.update_layout(xaxis_title='Contact Date', yaxis_title='Satisfaction Score', **dark_theme_settings, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Average Satisfaction Score by Interaction Type")
    satisfaction_by_type = crm_data.groupby('Interaction Type')['Satisfaction Score'].mean().reset_index()
    fig = go.Figure(data=[go.Bar(x=satisfaction_by_type['Interaction Type'], y=satisfaction_by_type['Satisfaction Score'], marker_color='#0078D7')])
    fig.update_layout(xaxis_title='Interaction Type', yaxis_title='Average Satisfaction Score', **dark_theme_settings, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

with tab_demo:
    demo_display = demo_data.copy()
    demo_display['Purchase Amount'] = demo_display['Purchase Amount'].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(demo_display, use_container_width=True)
    
    st.subheader("Age Group Distribution")
    age_counts = demo_data['Age Group'].value_counts().reset_index()
    age_counts.columns = ['Age Group', 'Count']
    fig = go.Figure(data=[go.Bar(x=age_counts['Age Group'], y=age_counts['Count'], marker_color='#0078D7')])
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Count', **dark_theme_settings, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Total Purchase Amount by Region")
    purchase_by_region = demo_data.groupby('Region')['Purchase Amount'].sum().reset_index()
    fig = go.Figure(data=[go.Bar(x=purchase_by_region['Region'], y=purchase_by_region['Purchase Amount'], marker_color='#0078D7')])
    fig.update_layout(xaxis_title='Region', yaxis_title='Total Purchase Amount (₹)', **dark_theme_settings, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

# Download button
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data as CSV", data=csv_data, file_name="filtered_sales_data.csv", mime="text/csv", help="Export the current filtered sales data to a CSV file for further analysis.")

# Footer
st.markdown("<p style='color: #A9A9A9; font-size: 12px; text-align: center;'>© 2025 Steel Insights | Crafted for smarter steel business decisions</p>", unsafe_allow_html=True)
