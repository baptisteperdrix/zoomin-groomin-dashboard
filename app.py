import os
import pandas as pd
import plotly.express as px
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

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

# Cash file expected columns
CASH_REQUIRED_COLS = ["Payment method", "Payment amount"]


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


def mobile_fee_pct_by_period(df: pd.DataFrame, period_order: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Period": period_order})

    tickets = df.groupby("Period").size().rename("Ticket count")
    if "Mobile fee count" in df.columns:
        mobile = df.groupby("Period")["Mobile fee count"].sum()
    else:
        mobile = pd.Series(dtype=float)

    out = base.copy()
    out["Ticket count"] = out["Period"].map(tickets).fillna(0).astype(int)
    out["Mobile fee count"] = out["Period"].map(mobile).fillna(0).astype(int)
    out["Mobile fee %"] = (out["Mobile fee count"] / out["Ticket count"].replace(0, pd.NA)).fillna(0)
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


# ---------------- FIGURE HELPERS + GOAL LINE ----------------
def add_goal_line(fig, goal_value, label="Goal", fmt=None):
    if goal_value is None:
        return fig
    try:
        g = float(goal_value)
    except Exception:
        return fig
    if g == 0:
        return fig

    fig.add_hline(
        y=g,
        line_dash="dash",
        line_width=2,
        annotation_text=f"{label}: {fmt(g) if fmt else g}",
        annotation_position="top left",
    )
    return fig


def bar_money(df: pd.DataFrame, x: str, y: str, palette: list[str], title: str, order: list[str], goal=None):
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
    fig = add_goal_line(fig, goal, "Goal", fmt=lambda v: f"${v:,.2f}")
    return fig


def bar_count(df: pd.DataFrame, x: str, y: str, palette: list[str], title: str, order: list[str], goal=None):
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
    fig = add_goal_line(fig, goal, "Goal", fmt=lambda v: f"{int(round(v))}")
    return fig


def bar_percent(df: pd.DataFrame, x: str, y: str, palette: list[str], title: str, order: list[str], goal_percent=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=palette,
        title=title,
        category_orders={x: order},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(texttemplate="%{y:.1%}", textposition="inside")
    fig.update_layout(showlegend=False)

    if goal_percent is not None:
        try:
            gp = float(goal_percent)
            if gp != 0:
                fig = add_goal_line(fig, gp / 100.0, "Goal", fmt=lambda v: f"{v*100:.1f}%")
        except Exception:
            pass

    return fig


def bar_grouped_money(df: pd.DataFrame, x: str, y: str, group: str, title: str, order: list[str], goal=None):
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
    fig = add_goal_line(fig, goal, "Goal", fmt=lambda v: f"${v:,.2f}")
    return fig


# ---------------- MAIN DASHBOARD PROCESS ----------------
def build_dashboard(
    file_obj,
    goal_revenue_w, goal_revenue_m,
    goal_avg_van_w, goal_avg_van_m,
    goal_groomer_w, goal_groomer_m,
    goal_pets_w, goal_pets_m,
    goal_apt_w, goal_apt_m,
    goal_mobile_pct_w, goal_mobile_pct_m,
    goal_discounts_w, goal_discounts_m,
):
    # outputs: status + 14 plots + unpaid_msg + unpaid_table = 17 outputs
    EMPTY_RET = ("Upload an Excel file to begin.",) + (None,) * 14 + ("", pd.DataFrame())

    if file_obj is None:
        return EMPTY_RET

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

        avg_ticket_weeks = avg_price_per_ticket(weeks_df, total_weeks)
        avg_ticket_months = avg_price_per_ticket(months_df, total_months)

        mobile_pct_weeks = mobile_fee_pct_by_period(weeks_df, weeks_order)
        mobile_pct_months = mobile_fee_pct_by_period(months_df, months_order)

        discounts_weeks = template_sum(weeks_df, weeks_order, "Discount", round2=True)
        discounts_months = template_sum(months_df, months_order, "Discount", round2=True)

        unpaid_note, unpaid_df = unpaid_table_through_last_week(df, last_week_end)
        if unpaid_df.empty and not unpaid_note:
            unpaid_msg = f"🎉 No unpaid appointments through {last_week_end.date()} — great job staying on top of collections!"
        else:
            unpaid_msg = unpaid_note or f"Includes appointments up to {last_week_end.date()} (end of last completed week)."

        status = (
            f"✅ Loaded {len(df):,} rows\n\n"
            f"- Anchor sale date: **{anchor.date()}**\n"
            f"- Last completed week ends: **{last_week_end.date()}**\n"
            f"- Last completed month ends: **{last_month_end.date()}**\n"
            f"- Van count (excluding '{UNASSIGNED_VAN_LABEL}'): **{van_count}**\n"
        )

        fig_total_weeks = bar_money(total_weeks, "Period", "Total collected", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal=goal_revenue_w)
        fig_total_months = bar_money(total_months, "Period", "Total collected", PALETTE_MONTHS, "Last 4 Months", months_order, goal=goal_revenue_m)

        fig_avg_van_weeks = bar_money(avg_weeks, "Period", "Avg per Van", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal=goal_avg_van_w)
        fig_avg_van_months = bar_money(avg_months, "Period", "Avg per Van", PALETTE_MONTHS, "Last 4 Months", months_order, goal=goal_avg_van_m)

        fig_groomer_weeks = bar_grouped_money(groomer_weeks, "Period", "Total collected", "Assigned staff", "Last 4 Weeks", weeks_order, goal=goal_groomer_w)
        fig_groomer_months = bar_grouped_money(groomer_months, "Period", "Total collected", "Assigned staff", "Last 4 Months", months_order, goal=goal_groomer_m)

        fig_pets_weeks = bar_count(pets_weeks, "Period", "Pet count", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal=goal_pets_w)
        fig_pets_months = bar_count(pets_months, "Period", "Pet count", PALETTE_MONTHS, "Last 4 Months", months_order, goal=goal_pets_m)

        fig_apt_weeks = bar_money(avg_ticket_weeks, "Period", "Avg price per ticket", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal=goal_apt_w)
        fig_apt_months = bar_money(avg_ticket_months, "Period", "Avg price per ticket", PALETTE_MONTHS, "Last 4 Months", months_order, goal=goal_apt_m)

        fig_mobile_weeks = bar_percent(mobile_pct_weeks, "Period", "Mobile fee %", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal_percent=goal_mobile_pct_w)
        fig_mobile_months = bar_percent(mobile_pct_months, "Period", "Mobile fee %", PALETTE_MONTHS, "Last 4 Months", months_order, goal_percent=goal_mobile_pct_m)

        fig_discounts_weeks = bar_money(discounts_weeks, "Period", "Discount", PALETTE_WEEKS, "Last 4 Weeks", weeks_order, goal=goal_discounts_w)
        fig_discounts_months = bar_money(discounts_months, "Period", "Discount", PALETTE_MONTHS, "Last 4 Months", months_order, goal=goal_discounts_m)

        return (
            status,
            fig_total_weeks, fig_total_months,
            fig_avg_van_weeks, fig_avg_van_months,
            fig_groomer_weeks, fig_groomer_months,
            fig_pets_weeks, fig_pets_months,
            fig_apt_weeks, fig_apt_months,
            fig_mobile_weeks, fig_mobile_months,
            fig_discounts_weeks, fig_discounts_months,
            unpaid_msg, unpaid_df,
        )

    except Exception as e:
        msg = f"❌ Error: {type(e).__name__}: {e}"
        return (msg,) + (None,) * 14 + ("", pd.DataFrame())


# ---------------- CASH FILE LOGIC (NO DROPDOWN) ----------------
def _is_cash_file(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return all(c in cols for c in CASH_REQUIRED_COLS)


def _date_like_cols_by_parsing(df: pd.DataFrame, min_parse_rate: float = 0.6) -> list[str]:
    candidates = []
    for c in df.columns:
        try:
            s = df[c]
            non_null = s.dropna()
            if non_null.empty:
                continue
            parsed = pd.to_datetime(non_null, errors="coerce")
            rate = parsed.notna().mean()
            if rate >= min_parse_rate:
                candidates.append(c)
        except Exception:
            continue

    preferred_order = [
        "Transaction datetime",
        "Transaction date",
        "Payment date",
        "Sale date",
    ]
    ordered = [c for c in preferred_order if c in candidates]
    for c in candidates:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _best_cash_date_col(df: pd.DataFrame) -> str | None:
    cols = _date_like_cols_by_parsing(df, min_parse_rate=0.6)
    return cols[0] if cols else None


def _parse_user_date(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, errors="raise").normalize()
    except Exception:
        return None


def cash_ui_state(file_obj):
    """
    Returns:
      cash_status_md, cash_container_update, cash_start_str, cash_end_str, cash_result_clear
    """
    if file_obj is None:
        return (
            "Upload a cash payments Excel file to calculate cash collected.",
            gr.update(visible=False),
            "",
            "",
            "",
        )

    try:
        df = pd.read_excel(file_obj.name, engine="openpyxl")
    except Exception as e:
        return (
            f"❌ Could not read file: {type(e).__name__}: {e}",
            gr.update(visible=False),
            "",
            "",
            "",
        )

    if not _is_cash_file(df):
        return (
            "⚠️ This doesn’t look like the cash payments file. "
            "Required columns: **Payment method**, **Payment amount**. Nothing will display until a valid cash file is uploaded.",
            gr.update(visible=False),
            "",
            "",
            "",
        )

    date_col = _best_cash_date_col(df)
    if not date_col:
        return (
            "✅ Cash file detected, but I couldn’t find any parseable date/datetime columns. "
            "Make sure there is a transaction/payment date column in the file.",
            gr.update(visible=True),
            "",
            "",
            "",
        )

    parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if parsed.empty:
        return (
            f"✅ Cash file detected, but **{date_col}** couldn’t be parsed as dates.",
            gr.update(visible=True),
            "",
            "",
            "",
        )

    min_d = parsed.min().date().isoformat()
    max_d = parsed.max().date().isoformat()

    msg = (
        f"✅ Cash file detected.\n\n"
        f"- Valid range: **{min_d}** to **{max_d}**\n\n"
        f"Enter dates as `YYYY-MM-DD`."
    )

    return (
        msg,
        gr.update(visible=True),
        min_d,
        max_d,
        "",
    )


def compute_cash_total(file_obj, start_str, end_str):
    if file_obj is None:
        return "Upload a payment transaction report first."

    try:
        df = pd.read_excel(file_obj.name, engine="openpyxl")
    except Exception as e:
        return f"❌ Could not read file: {type(e).__name__}: {e}"

    if not _is_cash_file(df):
        return "⚠️ Invalid payment transaction report. Required columns: **Payment method**, **Payment amount**."

    date_col = _best_cash_date_col(df)
    if not date_col:
        return "⚠️ Could not find a valid date/datetime column in this file."

    start = _parse_user_date(start_str)
    end = _parse_user_date(end_str)
    if start is None or end is None:
        return "⚠️ Please enter valid dates in **YYYY-MM-DD** format for both start and end."

    if end < start:
        return "⚠️ End date must be on or after start date."

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Payment method"] = df["Payment method"].astype(str).str.strip().str.casefold()
    df["Payment amount"] = pd.to_numeric(df["Payment amount"], errors="coerce").fillna(0)

    # inclusive end-of-day for the end date
    end_inclusive = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    sub = df.loc[
        df[date_col].notna() & (df[date_col] >= start) & (df[date_col] <= end_inclusive)
    ].copy()

    cash_total = sub.loc[sub["Payment method"].eq("cash"), "Payment amount"].sum()
    check_total = sub.loc[sub["Payment method"].eq("check"), "Payment amount"].sum()
    combined = cash_total + check_total

    # BIG, clean output (cash + checks)
    return (
        f"# Collections from {start.date()} to {end.date()}\n\n"
        f"## 💵 Cash: **${cash_total:,.2f}**\n"
        f"## 🧾 Checks: **${check_total:,.2f}**\n\n"
        f"### Total (Cash + Checks): **${combined:,.2f}**"
    )



# ---------------- GRADIO UI ----------------
with gr.Blocks(title="Zoomin Groomin Dashboard") as demo:
    gr.Markdown("# Zoomin Groomin Dashboard")

    file_in = gr.File(label="Upload Revenue Excel file (.xlsx)", file_types=[".xlsx", ".xls"])
    status_md = gr.Markdown()

    # ------- Revenue Tab -------
    with gr.Tab("Revenue"):
        with gr.Accordion("Goals (Revenue tab)", open=False):
            gr.Markdown("**Leave a goal blank to hide the goal line.** (A goal of 0 also hides the line.)")

            with gr.Row():
                goal_revenue_w = gr.Number(label="Goal: Collected Revenue (Weeks) $", value=None)
                goal_revenue_m = gr.Number(label="Goal: Collected Revenue (Months) $", value=None)

            with gr.Row():
                goal_avg_van_w = gr.Number(label="Goal: Avg Revenue / Van (Weeks) $", value=None)
                goal_avg_van_m = gr.Number(label="Goal: Avg Revenue / Van (Months) $", value=None)

            with gr.Row():
                goal_groomer_w = gr.Number(label="Goal: Revenue / Groomer (Weeks) $ — optional", value=None)
                goal_groomer_m = gr.Number(label="Goal: Revenue / Groomer (Months) $ — optional", value=None)

        gr.Markdown("## Collected Revenue")
        with gr.Row():
            rev_w = gr.Plot()
            rev_m = gr.Plot()

        gr.Markdown("## Collected Revenue / Van")
        with gr.Row():
            avgv_w = gr.Plot()
            avgv_m = gr.Plot()

        gr.Markdown("## Collected Revenue / Groomer")
        groomer_w = gr.Plot()
        groomer_m = gr.Plot()

    # ------- Ops KPIs Tab -------
    with gr.Tab("Ops KPIs"):
        with gr.Accordion("Goals (Ops KPIs tab)", open=False):
            gr.Markdown("**Leave a goal blank to hide the goal line.** (A goal of 0 also hides the line.)")

            with gr.Row():
                goal_pets_w = gr.Number(label="Goal: # Pets Serviced (Weeks)", value=None)
                goal_pets_m = gr.Number(label="Goal: # Pets Serviced (Months)", value=None)

            with gr.Row():
                goal_apt_w = gr.Number(label="Goal: Avg Price per Ticket (Weeks) $", value=None)
                goal_apt_m = gr.Number(label="Goal: Avg Price per Ticket (Months) $", value=None)

            with gr.Row():
                goal_mobile_pct_w = gr.Number(label="Goal: Mobile Fees % (Weeks) — enter like 25 for 25%", value=None)
                goal_mobile_pct_m = gr.Number(label="Goal: Mobile Fees % (Months) — enter like 25 for 25%", value=None)

            with gr.Row():
                goal_discounts_w = gr.Number(label="Goal: Discounts (Weeks) $", value=None)
                goal_discounts_m = gr.Number(label="Goal: Discounts (Months) $", value=None)

        gr.Markdown("## Pets Serviced")
        with gr.Row():
            pets_w = gr.Plot()
            pets_m = gr.Plot()

        gr.Markdown("## Average Price Per Ticket")
        with gr.Row():
            apt_w = gr.Plot()
            apt_m = gr.Plot()

        gr.Markdown("## Mobile Convenience Fees Applied")
        with gr.Row():
            mobile_w = gr.Plot()
            mobile_m = gr.Plot()

        gr.Markdown("## Discounts")
        with gr.Row():
            disc_w = gr.Plot()
            disc_m = gr.Plot()

    # ------- Unpaid Tab -------
    with gr.Tab("Unpaid Appointments"):
        unpaid_msg = gr.Markdown()
        unpaid_table = gr.Dataframe(interactive=False)

    # ------- Cash Collected Tab -------
    with gr.Tab("Cash Collected"):
        gr.Markdown(
            "Upload the **payment transaction report** file here."
        )
        cash_file = gr.File(label="Upload Payment Transaction Report (.xlsx)", file_types=[".xlsx", ".xls"])

        cash_status = gr.Markdown("Upload a cash payments Excel file to calculate cash collected.")
        cash_container = gr.Group(visible=False)

        with cash_container:
            with gr.Row():
                cash_start = gr.Textbox(label="Start date (YYYY-MM-DD)")
                cash_end = gr.Textbox(label="End date (YYYY-MM-DD)")
            cash_calc = gr.Button("Calculate Cash Collected")
            cash_result = gr.Markdown()

    # -------- DASHBOARD CALLBACK WIRING --------
    inputs_list = [
        file_in,
        goal_revenue_w, goal_revenue_m,
        goal_avg_van_w, goal_avg_van_m,
        goal_groomer_w, goal_groomer_m,
        goal_pets_w, goal_pets_m,
        goal_apt_w, goal_apt_m,
        goal_mobile_pct_w, goal_mobile_pct_m,
        goal_discounts_w, goal_discounts_m,
    ]

    outputs_list = [
        status_md,
        rev_w, rev_m,
        avgv_w, avgv_m,
        groomer_w, groomer_m,
        pets_w, pets_m,
        apt_w, apt_m,
        mobile_w, mobile_m,
        disc_w, disc_m,
        unpaid_msg, unpaid_table,
    ]

    file_in.change(fn=build_dashboard, inputs=inputs_list, outputs=outputs_list)

    for g in [
        goal_revenue_w, goal_revenue_m,
        goal_avg_van_w, goal_avg_van_m,
        goal_groomer_w, goal_groomer_m,
        goal_pets_w, goal_pets_m,
        goal_apt_w, goal_apt_m,
        goal_mobile_pct_w, goal_mobile_pct_m,
        goal_discounts_w, goal_discounts_m,
    ]:
        g.change(fn=build_dashboard, inputs=inputs_list, outputs=outputs_list)

    # -------- CASH CALLBACK WIRING --------
    cash_file.change(
        fn=cash_ui_state,
        inputs=[cash_file],
        outputs=[cash_status, cash_container, cash_start, cash_end, cash_result],
    )

    cash_calc.click(
        fn=compute_cash_total,
        inputs=[cash_file, cash_start, cash_end],
        outputs=[cash_result],
    )


# ---------------- FASTAPI APP (Render) ----------------
app = FastAPI()

def gradio_basic_auth(username: str, password: str) -> bool:
    return (
        username == os.environ.get("APP_USERNAME", "")
        and password == os.environ.get("APP_PASSWORD", "")
    )

app = gr.mount_gradio_app(app, demo, path="/", auth=gradio_basic_auth)