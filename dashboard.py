import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Zoomin Groomin Dashboard", layout="wide")

# --- CACHE FUNCTION ---
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

# --- TAB SWITCHER (GLITCH-FREE) ---
def tab_switch(labels):
    """Glitch-free, single-click persistent tabs in Streamlit."""

    def update_active_tab():
        st.session_state.active_tab = st.session_state.tabs_radio

    # Initialize on first run
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = labels[0]

    key = f"tabs_radio_{st.session_state.get('file_id', 'none')}"

    # Radio button controlled by Streamlit internal state
    st.radio(
        "Navigation",
        labels,
        horizontal=True,
        label_visibility="collapsed",
        key="tabs_radio",
        on_change=update_active_tab
    )

    # Make sure active_tab is synced with current radio value
    return st.session_state.get("active_tab", labels[0])


# --- FILE UPLOAD ---
st.title("🐾 Zoomin Groomin Dashboard")

uploaded_file = st.file_uploader("Upload your Appointment and Revenue Report", type=["xlsx"])

if uploaded_file:
    # Generate a unique ID for this upload
    new_file_id = hash(uploaded_file.name + str(uploaded_file.size))

    # Reset tab state when a new file is uploaded
    if st.session_state.get("file_id") != new_file_id:
        st.session_state.file_id = new_file_id
        st.session_state.active_tab = "📈 Revenue Over Time"
        if "tabs_radio" in st.session_state:
            del st.session_state["tabs_radio"]  # ensure clean reset

    data = load_data(uploaded_file)

    # --- CLEAN DATA ---
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    cols = ['Date', 'Staffs', 'Collected revenue', 'Pet name', 'Unpaid revenue',
            'First name', 'Last name', 'Primary number', 'Full address',
            'Time', 'Invoice ID']
    data = data[[c for c in cols if c in data.columns]]

    data['Collected revenue'] = pd.to_numeric(data['Collected revenue'], errors='coerce').fillna(0)
    data['Unpaid revenue'] = pd.to_numeric(data['Unpaid revenue'], errors='coerce').fillna(0)

    # --- METRICS ---
    total_revenue = data['Collected revenue'].sum()
    avg_revenue = data['Collected revenue'].mean()
    num_appointments = len(data)

    data['Pet count'] = data['Pet name'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    total_pets = data['Pet count'].sum()

    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("📊 Avg Revenue per Appointment", f"${avg_revenue:,.2f}")
    col3.metric("📅 Total Appointments", f"{num_appointments}")
    col4.metric("🐾 Pets Serviced", f"{total_pets}")

    st.markdown("---")

    # --- TAB NAVIGATION ---
    selected_tab = tab_switch([
        "📈 Revenue Over Time",
        "👥 Revenue by Staff",
        "🗓️ Appointments per Day",
        "🐾 Pets Serviced per Day",
        "💸 Unpaid Appointments",
        "❌ Cancellations"
    ])

    # --- TABS CONTENT ---
    if selected_tab == "📈 Revenue Over Time":
        revenue_time = data.groupby('Date', as_index=False)['Collected revenue'].sum()
        fig1 = px.line(revenue_time, x='Date', y='Collected revenue', title='Revenue Over Time')
        st.plotly_chart(fig1, use_container_width=True)

    elif selected_tab == "👥 Revenue by Staff":
        staff_revenue = (
            data.groupby('Staffs', as_index=False)['Collected revenue']
            .sum().sort_values('Collected revenue', ascending=False)
        )
        fig2 = px.bar(
            staff_revenue,
            x='Staffs',
            y='Collected revenue',
            title='Revenue by Staff',
            text_auto=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif selected_tab == "🗓️ Appointments per Day":
        appts_day = data.groupby('Date').size().reset_index(name='Appointments')
        fig3 = px.bar(appts_day, x='Date', y='Appointments', title='Appointments per Day')
        st.plotly_chart(fig3, use_container_width=True)

    elif selected_tab == "🐾 Pets Serviced per Day":
        pets_per_day = data.groupby('Date', as_index=False)['Pet count'].sum()
        fig4 = px.bar(pets_per_day, x='Date', y='Pet count', title='Pets Serviced per Day')
        st.plotly_chart(fig4, use_container_width=True)

    elif selected_tab == "💸 Unpaid Appointments":
        unpaid_df = data[data["Unpaid revenue"] > 0]
        if unpaid_df.empty:
            st.success("✅ All appointments are fully paid!")
        else:
            st.warning(f"⚠️ {len(unpaid_df)} unpaid appointment(s) found.")
            st.write("Below is the list of unpaid customers:")

            columns_to_show = [
                "First name", "Last name", "Primary number",
                "Full address", "Date", "Time",
                "Invoice ID", "Pet name", "Unpaid revenue"
                ]
            existing_columns = [col for col in columns_to_show if col in unpaid_df.columns]
            unpaid_table = unpaid_df[existing_columns].copy().sort_values("Date")

            st.dataframe(unpaid_table, use_container_width=True, hide_index=True)

            st.download_button(
                label="📥 Download Unpaid Appointments (CSV)",
                data=unpaid_table.to_csv(index=False),
                file_name="unpaid_appointments.csv",
                mime="text/csv"
            )
    
    elif selected_tab == "❌ Cancellations":
        st.subheader("❌ Cancelled Appointments")
        cancel_file = st.file_uploader("Upload the cancelled appointments Excel file", type=["xlsx"], key="cancel_upload")

        if cancel_file:
            cancel_data = load_data(cancel_file)
            num_cancelled = len(cancel_data)
            total_appointments = num_appointments + num_cancelled
            percent_cancelled = (num_cancelled / total_appointments) * 100 if total_appointments > 0 else 0

            color = "red" if percent_cancelled > 10 else "green"

            st.markdown(f"<h3>Total Cancelled Appointments: {num_cancelled}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='color:{color};'>Cancelled Appointments Rate: {percent_cancelled:.1f}%</h3>",
                unsafe_allow_html=True
            )
        else:
            st.info("📄 Upload your cancelled appointments report to calculate the cancellation rate.")

else:
    st.info("👆 Upload your Appointment and Revenue Report to begin.")