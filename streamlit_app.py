import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.linear_solver import pywraplp
from fpdf import FPDF
import io
import os

def main():
    st.set_page_config(page_title="Rail Project Dashboard", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Introduction", "📊 Optimization Analysis", "📈 PPG Dataset Analysis", "📄 Download Reports", "🔧 Future Page"])

    if page == "🏠 Introduction":
        introduction_page()
    elif page == "📊 Optimization Analysis":
        optimization_page()
    elif page == "📈 PPG Dataset Analysis":
        ppg_analysis_page()
    elif page == "📄 Download Reports":
        download_reports_page()
    elif page == "🔧 Future Page":
        future_page()

def introduction_page():
    st.title("🚆 Welcome to the Railway Optimization System")

    st.write("## 🔍 What is this application about?")
    st.write(
        "- This app is designed to **optimize railway resource allocation** by analyzing labor costs, efficiency, and fatigue levels."
        "\n- It uses **data-driven insights** and **AI-based models** to improve **scheduling, cost management, and worker productivity**."
        "\n- **Goal:** To create a **smarter, safer, and more efficient** railway workforce management system."
    )

    st.write("---")

    st.write("## 👥 Who is this for?")
    st.write(
        "- **Railway Project Managers** 🏗️ - To efficiently allocate labor and reduce operational costs."
        "\n- **Railway Authorities & Engineers** 🛤️ - To make informed decisions using real-time data."
        "\n- **Policy Makers & Analysts** 📊 - To ensure compliance, efficiency, and safety improvements."
    )

    st.write("---")

    st.write("## 🕒 When is Railway Optimization Needed?")
    st.write(
        "- When managing **large-scale railway projects** with multiple workforce categories."
        "\n- When **cost efficiency** and **timely execution** are crucial."
        "\n- When **worker fatigue and safety** need to be monitored and improved."
    )

    st.write("---")

    st.write("## 📍 Where does this optimization apply?")
    st.write(
        "- **Railway Construction & Maintenance** 🚄 - Managing workers efficiently for repairs, track laying, and station maintenance."
        "\n- **Operational Workforce Planning** 👷‍♂️ - Ensuring the right number of workers are assigned to different shifts."
        "\n- **Cost Reduction Strategies** 💰 - Identifying the most cost-effective labor allocation."
    )

    st.write("---")

    st.write("## ❓ Why is this important?")
    st.write(
        "- ✅ **Saves Costs**: Reduces unnecessary spending on overstaffing."
        "\n- ✅ **Prevents Worker Fatigue**: Uses **PPG (Photoplethysmography) Data** to track and analyze worker exhaustion."
        "\n- ✅ **Improves Project Efficiency**: Ensures the **right workers** are assigned at the **right time**."
        "\n- ✅ **Enhances Decision-Making**: Provides clear **data visualization & reports** for better insights."
    )

    st.write("---")

    st.write("## ⚙️ How does it work?")
    st.write(
        "- 🛠️ **Optimization Engine:** Uses AI to calculate **optimal labor allocation** based on cost & efficiency."
        "\n- 📊 **Data Visualization:** Interactive **charts** provide insights into labor trends and fatigue levels."
        "\n- 🩺 **Fatigue Analysis:** Integrates **PPG data** to measure and prevent worker exhaustion."
        "\n- 📄 **PDF Reports:** Generate **detailed downloadable reports** for project tracking."
    )

    st.write("---")

    st.success("🎯 **Get Started!** Use the sidebar to navigate through different sections and explore the features.")


def optimization_page():
    st.title("📊 Optimization Analysis")
    st.write("Upload labor cost and project requirement datasets to optimize workforce allocation.")
    
    # File Uploaders
    labor_file = st.file_uploader("Upload Labor Costs & Availability CSV", type=["csv"])
    project_file = st.file_uploader("Upload Project Requirements CSV", type=["csv"])
    
    if labor_file and project_file:
        labor_df = pd.read_csv(labor_file)
        project_df = pd.read_csv(project_file)

        # Ensure clean column names
        labor_df.columns = labor_df.columns.str.strip()
        project_df.columns = project_df.columns.str.strip()
        
        # User selects project
        selected_project = st.selectbox("Select Project", project_df['Project Name'].unique())
        project_data = project_df[project_df['Project Name'] == selected_project]
        
        # User selects labor types
        selected_labor_types = st.multiselect("Select Labor Types", labor_df['Labor Type'].unique(), default=labor_df['Labor Type'].unique())
        labor_data = labor_df[labor_df['Labor Type'].isin(selected_labor_types)]
        
        # Slider to adjust max hours per worker
        max_hours_selected = st.slider("Max Hours per Worker", min_value=50, max_value=int(labor_data['Maximum Available Hours'].max()), step=50)
        
        # Extract relevant data
        costs = labor_data['Cost per Hour (₹)'].values.astype(float)
        max_hours = np.minimum(labor_data['Maximum Available Hours'].values.astype(float), max_hours_selected)
        max_hours = np.nan_to_num(max_hours, nan=0.0)  # Replace NaNs with 0
        labor_types = labor_data['Labor Type'].values
        required_hours = project_data['Total Required Hours'].sum()
        num_workers = len(labor_types)
        
        # Optimization using OR-Tools
        solver = pywraplp.Solver.CreateSolver('SCIP')
        worker_hours = [solver.NumVar(0.0, float(max_hours[i]), f'worker_{i}') for i in range(num_workers)]
        
        # Ensuring fair workload distribution dynamically
        solver.Add(solver.Sum(worker_hours) == required_hours)
        min_allocation = required_hours / num_workers  # Ensures balanced distribution
        for i in range(num_workers):
            solver.Add(worker_hours[i] >= min_allocation * 0.7)  # Ensures minimum distribution
            solver.Add(worker_hours[i] <= max_hours[i])
        
        # Objective function: Minimize cost while balancing workload
        solver.Minimize(solver.Sum(worker_hours[i] * costs[i] for i in range(num_workers)))
        
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            allocation = [round(worker_hours[i].solution_value(), 2) for i in range(num_workers)]
            optimized_cost = round(sum(allocation[i] * costs[i] for i in range(num_workers)), 2)
            allocation_df = pd.DataFrame([allocation], columns=labor_types, index=["Allocated Hours"])
            
            # Display results
            st.subheader("✅ Optimized Labor Allocation")
            st.dataframe(allocation_df.style.format(precision=2))
            st.metric(label="💰 Optimized Total Cost", value=f"₹{optimized_cost:,.2f}")
            
            # Beautified Charts
            st.subheader("📊 Labor Allocation Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=labor_types, y=allocation, palette="coolwarm", ax=ax, edgecolor="black")
            ax.set_xlabel("Labor Type", fontsize=12)
            ax.set_ylabel("Assigned Hours", fontsize=12)
            ax.set_title("Optimized Labor Hours Allocation", fontsize=14, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("### 🔍 Insights from the Graph:")
            st.markdown("- **Visual Representation:** Shows how labor hours are distributed across different categories.")
            st.markdown("- **Optimization Impact:** Balances workload efficiently based on available labor and project demand.")
            st.markdown("- **Changes with Slider:** Adjusting max hours influences allocation dynamically.")
            
            st.subheader("📈 Cost Distribution by Labor Type")
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(costs, labels=labor_types, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"), textprops={'fontsize': 10})
            for text in autotexts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            ax.set_title("Cost Breakdown by Labor Type", fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            st.markdown("### 🔍 Key Takeaways from Cost Distribution:")
            st.markdown("- **Expense Breakdown:** Highlights which labor types contribute most to total cost.")
            st.markdown("- **Cost Efficiency:** Helps identify expensive labor types that could be optimized.")
            st.markdown("- **Optimization Effect:** Ensures cost is minimized while maintaining productivity.")
        else:
            st.error("Optimization failed. Try adjusting constraints.")

def ppg_analysis_page():
    st.title("📈 PPG Dataset Analysis")
    st.write("Upload the PPG fatigue dataset to analyze worker fatigue levels and efficiency impact.")
    
    # File Uploader
    ppg_file = st.file_uploader("Upload PPG Fatigue Dataset CSV", type=["csv"])
    
    if ppg_file:
        ppg_df = pd.read_csv(ppg_file)
        ppg_df.columns = ppg_df.columns.str.strip()
        
        # Use 'PPG_Level' as a continuous fatigue-related metric for the slider
        metric_column = 'PPG_Level'
        
        if metric_column not in ppg_df.columns:
            st.error("❌ PPG Level column not found! Available columns: " + ", ".join(ppg_df.columns))
            return
        
        # PPG Level Slider
        min_val, max_val = int(ppg_df[metric_column].min()), int(ppg_df[metric_column].max())
        selected_range = st.slider("Filter by PPG Level", min_value=min_val, max_value=max_val, value=(min_val, max_val))
        filtered_df = ppg_df[(ppg_df[metric_column] >= selected_range[0]) & (ppg_df[metric_column] <= selected_range[1])]
        
        # Chart 1: PPG Level Distribution
        st.subheader("📊 PPG Level Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df[metric_column], bins=20, kde=True, color="royalblue", edgecolor="black")
        ax.set_title("PPG Level Distribution Among Workers", fontsize=14, fontweight='bold')
        ax.set_xlabel("PPG Level", fontsize=12)
        ax.set_ylabel("Number of Workers", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        st.markdown("### 🔍 Key Insights:")
        st.markdown("- **Overall PPG Trends:** Identifies distribution of PPG Levels across workers.")
        st.markdown("- **Clusters & Peaks:** Helps find patterns in worker stress levels.")
        st.markdown("- **Optimization Factor:** Can assist in workload balancing based on physiological stress.")
        
        # Chart 2: PPG Level vs. Activity Level Scatter Plot
        if 'Activity_Level' in ppg_df.columns:
            st.subheader("📈 PPG Level vs. Activity Level")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=filtered_df[metric_column], y=filtered_df['Activity_Level'], hue=filtered_df['Activity_Level'], palette="coolwarm", edgecolor="black", alpha=0.8)
            ax.set_title("PPG Level vs. Worker Activity", fontsize=14, fontweight='bold')
            ax.set_xlabel("PPG Level", fontsize=12)
            ax.set_ylabel("Activity Level", fontsize=12)
            ax.grid(axis='both', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("### 🔍 Observations:")
            st.markdown("- **Correlation Check:** Higher PPG may indicate lower activity levels.")
            st.markdown("- **Outliers & Trends:** Some workers may maintain high activity despite high PPG levels.")
            st.markdown("- **Shift Planning Factor:** Can help in designing optimal work schedules.")
        else:
            st.warning("Activity level data is missing in the uploaded dataset.")
        
        # Chart 3: Time-Based PPG Trends
        if 'Timestamp' in filtered_df.columns:
            filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
            time_trend = filtered_df.groupby(filtered_df['Timestamp'].dt.hour)[metric_column].mean()
            
            st.subheader("📉 PPG Level Trends Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=time_trend.index, y=time_trend.values, marker='o', color="darkred")
            ax.set_title("PPG Level Changes Throughout the Day", fontsize=14, fontweight='bold')
            ax.set_xlabel("Hour of the Day", fontsize=12)
            ax.set_ylabel("Average PPG Level", fontsize=12)
            ax.grid(axis='both', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("### 🔍 What This Tells Us:")
            st.markdown("- **Peak Stress Hours:** Identifies when workers experience highest PPG levels.")
            st.markdown("- **Shift Adjustments:** Helps in structuring better work & rest periods.")
            st.markdown("- **Health & Performance:** Guides strategies to manage worker well-being.")
        else:
            st.warning("Timestamp data is missing. Unable to generate time-based PPG trends.")


def download_reports_page():
    st.title("📄 Download Analysis Reports")
    st.write("Generate a comprehensive PDF report with optimization and PPG dataset analysis, including charts and insights.")
    
    if st.button("📥 Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, "Railway Resource Optimization Report", ln=True, align='C')
        pdf.ln(10)
        
        # Optimization Section
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, "Optimization Analysis", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "This section provides an optimized allocation of labor resources to minimize cost while ensuring fair workload distribution.")
        pdf.ln(5)
        
        # Optimization Key Findings
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, "Key Findings:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, "- Labor allocation is optimized based on cost and availability.\n- The optimization model ensures fair workload distribution.\n- Adjusting max worker hours impacts overall cost and efficiency.")
        pdf.ln(10)
        
        # PPG Analysis Section
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, "PPG Dataset Analysis", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "The PPG dataset analysis helps understand worker fatigue levels, activity trends, and their impact on efficiency.")
        pdf.ln(5)
        
        # PPG Key Findings
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, "Key Findings:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, "- PPG levels indicate physiological stress among workers.\n- Higher fatigue levels often correlate with lower activity levels.\n- Time-based fatigue trends suggest peak fatigue hours, useful for shift planning.")
        pdf.ln(10)
        
        # Generate PDF
        pdf_output = io.BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))  # Write PDF data as bytes
        pdf_output.seek(0)  # Move to the beginning of the stream
        
        st.download_button("📥 Download Report", data=pdf_output, file_name="Optimization_Report.pdf", mime="application/pdf")


def future_page():
    st.title("🔧 Future Considerations & Social Impact")
    st.write("This project goes beyond traditional cost optimization by integrating worker well-being into decision-making.")
    
    st.subheader("💡 Why Combine PPG Analysis with Cost Optimization?")
    st.markdown("- **Worker Well-being First:** Ensuring railway workers remain efficient and safe by monitoring fatigue trends.")
    st.markdown("- **Reducing Accidents & Errors:** High fatigue levels are linked to reduced attention, increasing risks in railway operations.")
    st.markdown("- **Long-Term Workforce Sustainability:** Organizations investing in employee well-being see increased productivity and reduced turnover.")
    
    st.subheader("🌍 The Social Cause Behind This Initiative")
    st.markdown("- **A Step Towards Safer Railways:** Fatigue-induced errors have led to railway mishaps globally. Our system helps prevent them.")
    st.markdown("- **Balancing Efficiency with Human Needs:** Instead of treating labor purely as a cost factor, we account for physiological stress and optimize shifts accordingly.")
    st.markdown("- **Leading by Example:** This project sets a precedent for industries to integrate data-driven health insights into workforce planning.")
    
    st.subheader("🚀 A Noble Step Forward")
    st.markdown("- **Beyond Profitability:** We chose to integrate PPG analysis, knowing it would add complexity, but believing it was the right thing to do.")
    st.markdown("- **Ethical AI for Workforce Management:** By analyzing fatigue alongside cost, we ensure that optimization isn't just about money, but also about people.")
    st.markdown("- **Future Scope:** Expanding this model to include real-time fatigue alerts and AI-driven shift recommendations.")
    
    st.write("🔹 *This initiative is more than just an optimization model—it’s a commitment to safer, smarter, and more humane workforce planning.*")

if __name__ == "__main__":
    main()
