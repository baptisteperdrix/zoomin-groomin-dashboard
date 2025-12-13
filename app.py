import pandas as pd
import plotly.express as px
import gradio as gr
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets


# ---------------- CONFIG ----------------
VAN_LABEL_MAP = {
    "OG dawg 1(8502)": "OG Dawg #1",
    "Blue Betty #2(9534)": "Blue Betty #2",
    "Woof wagon #3(10272)": "Woof Wagon #3",
    "Van 4(11385)": "Van #4",
}

PALETTE_WEEKS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
PALETTE_MONTHS = ["#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]

UNASSIGNED_VAN_LABEL = "Unassigned van"
UNASSIGNED_GROOMER_LABEL = "Unassigned Groomer"
MOBILE_FEE_PHRASE = "Mobile Fee Convenience Charge"


# ---------------- DATA PREP ----------------
def _to_num(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Sale date" not in df.columns:
        raise ValueError("Missing required column: 'Sale date'")
    if "Total collected" not in df.columns:
        raise ValueError("Missing required column: 'Total collected'")

    df["Sale date"] = pd.to_datetime(df["Sale date"], errors="coerce")
    df = df[df["Sale date"].notna()].copy()

    _to_num(df, "Total collected")
    _to_num(df, "Total unpaid")
    _to_num(df, "Add-on net sales")
    _to_num(df, "Discount")

    # Van normalization
    if "Van" in df.columns:
        df["Van"] = (
            df["Van"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace(VAN_LABEL_MAP)
        )
        df["Van"] = df["Van"].where(df["Van"].ne(""), other=UNASSIGNED_VAN_LABEL)
        df.loc[df["Van"].str.lower().isin(["unassigned", "unassigned van"]), "Van"] = UNASSIGNED_VAN_LABEL
    else:
        df["Van"] = UNASSIGNED_VAN_LABEL

    # Groomer normalization
    if "Assigned staff" in df.columns:
        df["Assigned staff"] = (
            df["Assigned staff"]
            .fillna(UNASSIGNED_GROOMER_LABEL)
            .astype(str)
            .str.strip()
        )
        df.loc[df["Assigned staff"].eq(""), "Assigned staff"] = UNASSIGNED_GROOMER_LABEL
    else:
        df["Assigned staff"] = UNASSIGNED_GROOMER_LABEL

    return df


def add_pet_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Pet name" not in df.columns:
        df["Pet count"] = 0
        return df

    def count_pets(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0
        return str(val).count(",") + 1

    df["Pet count"] = df["Pet name"].apply(count_pets)
    return df


def add_mobile_fee_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Add-on" not in df.columns:
        df["Mobile fee count"] = 0
        return df

    df["Mobile fee count"] = (
        df["Add-on"].astype(str).str.contains(MOBILE_FEE_PHRASE, case=False, na=False).astype(int)
    )
    return df


# ---------------- COMPLETED WEEKS ----------------
def last_completed_saturday(anchor: pd.Timestamp) -> pd.Timestamp:
    wd = anchor.weekday()  # Mon=0 ... Sun=6
    if wd == 5:  # Saturday
        return anchor
    return anchor - pd.Timedelta(days=(wd - 5) % 7)


def week_end_saturday(d: pd.Series) -> pd.Series:
    offset = (5 - d.dt.weekday) % 7
    return d.dt.normalize() + pd.to_timedelta(offset, unit="D")


def filter_last_4_completed_weeks(df: pd.DataFrame, anchor: pd.Timestamp):
    end0 = last_completed_saturday(anchor)
    week_ends = [end0 - pd.Timedelta(days=7 * i) for i in range(4)]   # most recent first
    week_starts = [we - pd.Timedelta(days=6) for we in week_ends]     # Sunday starts

    labels = [f"Week of {ws.date().isoformat()}" for ws in week_starts]  # most recent first
    end_to_label = dict(zip([we.date() for we in week_ends], labels))

    sub = df.copy()
    sub["WeekEnd"] = week_end_saturday(sub["Sale date"])
    sub = sub[sub["WeekEnd"].dt.date.isin(end_to_label.keys())].copy()
    sub["Period"] = sub["WeekEnd"].dt.date.map(end_to_label)

    plot_order = labels[::-1]   # oldest -> newest
    return sub, plot_order, end0


# ---------------- COMPLETED MONTHS ----------------
def last_completed_month_end(anchor: pd.Timestamp) -> pd.Timestamp:
    anchor = anchor.normalize()
    current_month_end = (anchor + pd.offsets.MonthEnd(0)).normalize()
    if anchor != current_month_end:
        return (anchor.replace(day=1) - pd.Timedelta(days=1)).normalize()
    return current_month_end


def filter_last_4_completed_months(df: pd.DataFrame, anchor: pd.Timestamp):
    end0 = last_completed_month_end(anchor)

    ends = [end0]
    for _ in range(3):
        ends.append((ends[-1].replace(day=1) - pd.Timedelta(days=1)).normalize())

    labels = [pd.Timestamp(e).strftime("%B %Y") for e in ends]  # most recent first
    end_to_label = dict(zip([e.date() for e in ends], labels))

    sub = df.copy()
    sub["MonthEnd"] = (sub["Sale date"] + pd.offsets.MonthEnd(0)).dt.normalize()
    sub = sub[sub["MonthEnd"].dt.date.isin(end_to_label.keys())].copy()
    sub["Period"] = sub["MonthEnd"].dt.date.map(end_to_label)

    plot_order = labels[::-1]  # oldest -> newest
    return sub, plot_order, end0


# ---------------- AGGREGATIONS ----------------
def template_sum(df: pd.DataFrame, period_order: list[str], col: str, round2: bool = False) -> pd.DataFrame:
    base = pd.DataFrame({"Period": period_order})
    if col not in df.columns:
        out = base.copy()
        out[col] = 0
        return out

    grouped = df.groupby("Period", as_index=False)[col].sum()
    out = base.merge(grouped, on="Period", how="left").fillna({col: 0})
    if round2:
        out[col] = out[col].round(2)
    return out


def total_by_period(df: pd.DataFrame, period_order: list[str]) -> pd.DataFrame:
    return template_sum(df, period_order, "Total collected", round2=True)


def groomer_by_period(df: pd.DataFrame, total_df: pd.DataFrame) -> pd.DataFrame:
    groomer = (
        df.groupby(["Period", "Assigned staff"], as_index=False)["Total collected"]
        .sum()
        .round(2)
    )
    diff = (
        groomer.groupby("Period")["Total collected"].sum()
        - total_df.set_index("Period")["Total collected"]
    ).abs().max()
    assert diff < 0.01, f"Groomer totals do not reconcile (diff ${diff:,.2f})"
    return groomer


def fleet_van_count(df: pd.DataFrame) -> int:
    return len([v for v in df["Van"].unique() if v != UNASSIGNED_VAN_LABEL])


def avg_per_van(total_df: pd.DataFrame, van_count: int) -> pd.DataFrame:
    out = total_df.copy()
    out["Avg per Van"] = out["Total collected"] / van_count if van_count else 0
    diff = (out["Avg per Van"] * van_count - out["Total collected"]).abs().max()
    assert diff < 0.01, f"Avg/Van does not reconcile (diff ${diff:,.2f})"
    return out


def avg_price_per_ticket(df: pd.DataFrame, total_df: pd.DataFrame) -> pd.DataFrame:
    tickets = (
        df.groupby("Period", as_index=False)
        .size()
        .rename(columns={"size": "Ticket count"})
    )
    out = total_df.merge(tickets, on="Period", how="left").fillna({"Ticket count": 0})
    out["Avg price per ticket"] = (out["Total collected"] / out["Ticket count"].replace(0, pd.NA)).fillna(0)

    diff = (out["Avg price per ticket"] * out["Ticket count"] - out["Total collected"]).abs().max()
    assert diff < 0.01, f"Avg price per ticket does not reconcile (diff ${diff:,.2f})"
    return out


def unpaid_table_through_last_week(df: pd.DataFrame, last_week_end: pd.Timestamp) -> tuple[str, pd.DataFrame]:
    required = ["Client name", "Client email", "Pet name", "Sale date", "Total unpaid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"⚠️ Missing columns for unpaid table: {', '.join(missing)}", pd.DataFrame(columns=required)

    unpaid = df.loc[(df["Sale date"] <= last_week_end) & (df["Total unpaid"] > 0), required].copy()
    unpaid = unpaid.sort_values("Sale date", ascending=False)
    if not unpaid.empty:
        unpaid["Sale date"] = unpaid["Sale date"].dt.date
    return "", unpaid


# ---------------- FIGURE HELPERS ----------------
def bar_money(df: pd.DataFrame, x: str, y: str, palette: list[str], title: str, order: list[str]):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=palette,
        title=title,
        category_orders={x: order},
    )
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_traces(texttemplate="$%{y:,.2f}", textposition="inside")
    fig.update_layout(showlegend=False)
    return fig


def bar_count(df: pd.DataFrame, x: str, y: str, palette: list[str], title: str, order: list[str]):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=palette,
        title=title,
        category_orders={x: order},
    )
    fig.update_traces(texttemplate="%{y}", textposition="inside")
    fig.update_layout(showlegend=False)
    return fig


def bar_grouped_money(df: pd.DataFrame, x: str, y: str, group: str, title: str, order: list[str]):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=group,
        barmode="group",
        title=title,
        category_orders={x: order},
    )
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_traces(texttemplate="$%{y:,.2f}", textposition="inside")
    return fig


# ---------------- MAIN PROCESS ----------------
def build_dashboard(file_obj):
    if file_obj is None:
        return ("Upload an Excel file to begin.", None, None, None, None, None, None, None, None, None, None, None)

    try:
        raw = pd.read_excel(file_obj.name, engine="openpyxl")
        df = prep_df(raw)
        df = add_pet_count(df)
        df = add_mobile_fee_flag(df)

        anchor = df["Sale date"].max().normalize()

        weeks_df, weeks_order, last_week_end = filter_last_4_completed_weeks(df, anchor)
        months_df, months_order, last_month_end = filter_last_4_completed_months(df, anchor)

        total_weeks = total_by_period(weeks_df, weeks_order)
        total_months = total_by_period(months_df, months_order)

        groomer_weeks = groomer_by_period(weeks_df, total_weeks)
        groomer_months = groomer_by_period(months_df, total_months)

        van_count = fleet_van_count(df)
        avg_weeks = avg_per_van(total_weeks, van_count)
        avg_months = avg_per_van(total_months, van_count)

        pets_weeks = template_sum(weeks_df, weeks_order, "Pet count", round2=False)
        pets_months = template_sum(months_df, months_order, "Pet count", round2=False)

        mobile_weeks = template_sum(weeks_df, weeks_order, "Mobile fee count", round2=False)
        mobile_months = template_sum(months_df, months_order, "Mobile fee count", round2=False)

        addons_weeks = template_sum(weeks_df, weeks_order, "Add-on net sales", round2=True)
        addons_months = template_sum(months_df, months_order, "Add-on net sales", round2=True)

        discounts_weeks = template_sum(weeks_df, weeks_order, "Discount", round2=True)
        discounts_months = template_sum(months_df, months_order, "Discount", round2=True)

        avg_ticket_weeks = avg_price_per_ticket(weeks_df, total_weeks)
        avg_ticket_months = avg_price_per_ticket(months_df, total_months)

        unpaid_note, unpaid_df = unpaid_table_through_last_week(df, last_week_end)

        status = (
            f"✅ Loaded {len(df):,} rows\n\n"
            f"- Anchor sale date: **{anchor.date()}**\n"
            f"- Last completed week ends: **{last_week_end.date()}**\n"
            f"- Last completed month ends: **{last_month_end.date()}**\n"
            f"- Van count (excluding '{UNASSIGNED_VAN_LABEL}'): **{van_count}**\n"
        )
        if unpaid_note:
            status += f"\n{unpaid_note}\n"

        # Figures
        fig_total_weeks = bar_money(total_weeks, "Period", "Total collected", PALETTE_WEEKS, "Collected Revenue — Last 4 Weeks", weeks_order)
        fig_total_months = bar_money(total_months, "Period", "Total collected", PALETTE_MONTHS, "Collected Revenue — Last 4 Months", months_order)

        fig_avg_van_weeks = bar_money(avg_weeks, "Period", "Avg per Van", PALETTE_WEEKS, "Avg Collected Revenue / Van — Last 4 Weeks", weeks_order)
        fig_avg_van_months = bar_money(avg_months, "Period", "Avg per Van", PALETTE_MONTHS, "Avg Collected Revenue / Van — Last 4 Months", months_order)

        fig_groomer_weeks = bar_grouped_money(groomer_weeks, "Period", "Total collected", "Assigned staff", "Collected Revenue / Groomer — Last 4 Weeks", weeks_order)
        fig_groomer_months = bar_grouped_money(groomer_months, "Period", "Total collected", "Assigned staff", "Collected Revenue / Groomer — Last 4 Months", months_order)

        fig_pets_weeks = bar_count(pets_weeks, "Period", "Pet count", PALETTE_WEEKS, "# of Pets Serviced — Last 4 Weeks", weeks_order)
        fig_pets_months = bar_count(pets_months, "Period", "Pet count", PALETTE_MONTHS, "# of Pets Serviced — Last 4 Months", months_order)

        fig_avg_ticket_weeks = bar_money(avg_ticket_weeks, "Period", "Avg price per ticket", PALETTE_WEEKS, "Avg Price per Ticket — Last 4 Weeks", weeks_order)
        fig_avg_ticket_months = bar_money(avg_ticket_months, "Period", "Avg price per ticket", PALETTE_MONTHS, "Avg Price per Ticket — Last 4 Months", months_order)

        fig_mobile_weeks = bar_count(mobile_weeks, "Period", "Mobile fee count", PALETTE_WEEKS, "Mobile Convenience Fees Applied — Last 4 Weeks", weeks_order)
        fig_mobile_months = bar_count(mobile_months, "Period", "Mobile fee count", PALETTE_MONTHS, "Mobile Convenience Fees Applied — Last 4 Months", months_order)

        fig_addons_weeks = bar_money(addons_weeks, "Period", "Add-on net sales", PALETTE_WEEKS, "Add-on Revenue — Last 4 Weeks", weeks_order)
        fig_addons_months = bar_money(addons_months, "Period", "Add-on net sales", PALETTE_MONTHS, "Add-on Revenue — Last 4 Months", months_order)

        fig_discounts_weeks = bar_money(discounts_weeks, "Period", "Discount", PALETTE_WEEKS, "Discounts — Last 4 Weeks", weeks_order)
        fig_discounts_months = bar_money(discounts_months, "Period", "Discount", PALETTE_MONTHS, "Discounts — Last 4 Months", months_order)

        # Unpaid message
        if unpaid_df.empty and not unpaid_note:
            unpaid_msg = f"🎉 No unpaid appointments through {last_week_end.date()} — great job staying on top of collections!"
        else:
            unpaid_msg = f"Includes appointments up to {last_week_end.date()} (end of last completed week)."

        # Return outputs in exact order defined in UI
        return (
            status,
            fig_total_weeks, fig_total_months,
            fig_avg_van_weeks, fig_avg_van_months,
            fig_groomer_weeks, fig_groomer_months,
            fig_pets_weeks, fig_pets_months,
            fig_avg_ticket_weeks, fig_avg_ticket_months,
            fig_mobile_weeks, fig_mobile_months,
            fig_addons_weeks, fig_addons_months,
            fig_discounts_weeks, fig_discounts_months,
            unpaid_msg, unpaid_df
        )

    except Exception as e:
        msg = f"❌ Error: {type(e).__name__}: {e}"
        # Return Nones for figs/dfs
        return (msg,) + (None,) * 18


# ---------------- GRADIO UI ----------------
with gr.Blocks(title="Zoomin Groomin Dashboard") as demo:
    gr.Markdown("# Zoomin Groomin Dashboard — Revenue")
    file_in = gr.File(label="Upload Excel file (.xlsx)", file_types=[".xlsx", ".xls"])

    status = gr.Markdown()

    with gr.Tab("Revenue"):
        with gr.Row():
            rev_w = gr.Plot()
            rev_m = gr.Plot()
        with gr.Row():
            avgv_w = gr.Plot()
            avgv_m = gr.Plot()
        gr.Markdown("## Collected Revenue / Groomer")
        groomer_w = gr.Plot()
        groomer_m = gr.Plot()

    with gr.Tab("Ops KPIs"):
        with gr.Row():
            pets_w = gr.Plot()
            pets_m = gr.Plot()
        with gr.Row():
            apt_w = gr.Plot()
            apt_m = gr.Plot()
        with gr.Row():
            mobile_w = gr.Plot()
            mobile_m = gr.Plot()
        with gr.Row():
            addons_w = gr.Plot()
            addons_m = gr.Plot()
        with gr.Row():
            disc_w = gr.Plot()
            disc_m = gr.Plot()

    with gr.Tab("Unpaid Appointments"):
        unpaid_msg = gr.Markdown()
        unpaid_table = gr.Dataframe(interactive=False)

    file_in.change(
        fn=build_dashboard,
        inputs=file_in,
        outputs=[
            status,
            rev_w, rev_m,
            avgv_w, avgv_m,
            groomer_w, groomer_m,
            pets_w, pets_m,
            apt_w, apt_m,
            mobile_w, mobile_m,
            addons_w, addons_m,
            disc_w, disc_m,
            unpaid_msg, unpaid_table
        ],
    )

# ---------------- BASIC AUTH ----------------
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.environ.get("APP_USERNAME")
    correct_password = os.environ.get("APP_PASSWORD")

    if not correct_username or not correct_password:
        raise HTTPException(status_code=500, detail="Auth not configured")

    is_user = secrets.compare_digest(credentials.username, correct_username)
    is_pass = secrets.compare_digest(credentials.password, correct_password)

    if not (is_user and is_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username

# ---------------- FASTAPI APP ----------------
app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# Mount Gradio with auth
app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    auth=authenticate
)
