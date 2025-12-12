import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title="Mobile Grooming KPI Dashboard", layout="wide")

# ---- Load auth config from Streamlit Secrets ----
auth_cfg = st.secrets["auth"]

credentials = auth_cfg["credentials"]
cookie_name = auth_cfg["cookie_name"]
cookie_key = auth_cfg["cookie_key"]
cookie_expiry_days = int(auth_cfg["cookie_expiry_days"])

authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name=cookie_name,
    cookie_key=cookie_key,
    cookie_expiry_days=cookie_expiry_days,
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("Username/password is incorrect.")
    st.stop()

if authentication_status is None:
    st.info("Please enter your username and password.")
    st.stop()

# ---- Authenticated area ----
authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Logged in as: {name} ({username})")

st.title("📊 Zoomin Groomin KPI Dashboard")

# Example: file uploader (you’ll expand this)
files = st.file_uploader(
    "Upload Excel file(s)",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)

if files:
    st.success(f"Received {len(files)} file(s). Ready to process.")
