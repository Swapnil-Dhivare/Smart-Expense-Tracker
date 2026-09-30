"""Smart Expense Categorization + Spending Prediction.

Upload transactions -> an ML classifier categorizes them -> trends, an
ensemble forecast, IsolationForest anomaly flags, and a budget check.
A Model Insights tab exposes cross-validation accuracy, a confusion matrix,
and per-category token weights; the Categorization tab lets you correct a
misclassified row and retrain the model on the spot.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from categorizer import CATEGORIES, ExpenseCategorizer
from predictor import (
    budget_status,
    category_distribution,
    detect_unusual,
    monthly_totals,
    predict_next_month,
)
from theme import (
    CATEGORY_COLORS,
    CUSTOM_CSS,
    MUTED_INK,
    SECONDARY_INK,
    SEQUENTIAL_BLUES,
    STATUS_COLORS,
    banner_html,
    category_label,
    kpi_card_html,
    themed,
)

st.set_page_config(page_title="Expense Intelligence", page_icon="💠", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_categorizer() -> ExpenseCategorizer:
    return ExpenseCategorizer()


def load_transactions(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/sample_transactions.csv")

    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        if col in ("merchant", "description", "narration", "details"):
            rename_map[col] = "merchant"
        elif col in ("amount", "amt", "value"):
            rename_map[col] = "amount"
        elif col in ("date", "txn_date", "transaction_date"):
            rename_map[col] = "date"
    df = df.rename(columns=rename_map)

    missing = {"merchant", "amount"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    if "date" not in df.columns:
        df["date"] = pd.Timestamp.today().normalize()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.markdown("### 💠 Expense Intelligence")
    st.caption("ML categorization · ensemble forecasting · anomaly detection")
    st.markdown("---")
    uploaded_file = st.file_uploader("Transactions CSV", type=["csv"])
    if uploaded_file is None:
        st.caption("No file uploaded — showing sample data.")
    st.markdown("---")
    budget = st.number_input("Monthly budget (₹)", min_value=0, value=15000, step=500)
    contamination = st.slider(
        "Anomaly sensitivity", 0.03, 0.20, 0.08, 0.01,
        help="Expected fraction of transactions flagged as anomalous by the IsolationForest model.",
    )
    st.markdown("---")
    categorizer = get_categorizer()
    st.caption(f"Model corrections learned: **{categorizer.correction_count}**")
    if st.button("↺ Reset model to defaults"):
        get_categorizer.clear()
        st.session_state.pop("editor", None)
        st.rerun()

try:
    raw_df = load_transactions(uploaded_file)
except ValueError as e:
    st.error(str(e))
    st.stop()

if raw_df.empty:
    st.warning("No valid transactions found in the uploaded file.")
    st.stop()

predictions = categorizer.predict(raw_df["merchant"].astype(str).tolist())
raw_df["category"] = [c for c, _ in predictions]
raw_df["confidence"] = [round(c, 3) for _, c in predictions]

st.markdown(
    "<h1>💠 Expense Intelligence</h1>"
    "<p style='margin-top:-10px;'>ML-driven categorization, ensemble spending forecast, "
    "and anomaly detection.</p>",
    unsafe_allow_html=True,
)

tab_overview, tab_categorize, tab_forecast, tab_anomalies, tab_model = st.tabs(
    ["📊 Overview", "🏷️ Categorization", "📈 Forecast", "🚨 Anomalies", "🧠 Model Insights"]
)

# --------------------------------------------------------------- Overview --
with tab_overview:
    dist = category_distribution(raw_df)
    result = predict_next_month(raw_df)
    status = budget_status(result.predicted_amount, budget)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card_html("Total Spend", f"₹{raw_df['amount'].sum():,.0f}",
                          f"{len(raw_df)} transactions"),
            unsafe_allow_html=True,
        )
    with c2:
        top_cat = dist.index[0]
        st.markdown(
            kpi_card_html("Top Category", category_label(top_cat),
                          f"₹{dist.iloc[0]:,.0f}"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card_html("Predicted Next Month", f"₹{result.predicted_amount:,.0f}",
                          f"±₹{(result.ci_upper - result.predicted_amount):,.0f} (95% band)"),
            unsafe_allow_html=True,
        )
    with c4:
        avg_conf = raw_df["confidence"].mean()
        st.markdown(
            kpi_card_html("Avg. Model Confidence", f"{avg_conf:.0%}",
                          "across all categorized transactions"),
            unsafe_allow_html=True,
        )

    st.write("")
    if status["status"] == "over":
        st.markdown(banner_html("over",
            f"⚠️ <b>Predicted spend exceeds budget</b> — ₹{result.predicted_amount:,.0f} projected "
            f"against a ₹{budget:,.0f} budget ({status['pct_used']:.0f}% used, over by "
            f"₹{status['over_by']:,.0f})."), unsafe_allow_html=True)
    elif status["status"] == "warning":
        st.markdown(banner_html("warning",
            f"⚠️ <b>Approaching budget</b> — predicted spend is at {status['pct_used']:.0f}% "
            f"of your ₹{budget:,.0f} budget."), unsafe_allow_html=True)
    elif status["status"] == "ok":
        st.markdown(banner_html("ok",
            f"✅ <b>On track</b> — predicted spend is {status['pct_used']:.0f}% of your "
            f"₹{budget:,.0f} budget."), unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=dist.values, names=dist.index, hole=0.55,
            color=dist.index, color_discrete_map=CATEGORY_COLORS,
            title="Spending by Category",
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(themed(fig_pie), width='stretch')
    with col2:
        totals = monthly_totals(raw_df)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=totals.index.strftime("%b %Y"), y=totals.values,
            marker_color="#3987e5", name="Monthly spend",
        ))
        fig_trend.update_layout(title="Monthly Spending Trend",
                                 yaxis_title="₹", showlegend=False)
        st.plotly_chart(themed(fig_trend), width='stretch')

# ----------------------------------------------------------- Categorization --
with tab_categorize:
    st.subheader("Categorized Transactions")
    st.caption(
        "Correct a category below and click **Apply corrections & retrain** — "
        "the classifier folds your fix into its training set and refits immediately."
    )

    display_df = raw_df[["date", "merchant", "amount", "category", "confidence"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["confidence"] = (display_df["confidence"] * 100).round(0)

    edited_df = st.data_editor(
        display_df,
        column_config={
            "category": st.column_config.SelectboxColumn("category", options=CATEGORIES),
            "confidence": st.column_config.NumberColumn("confidence", format="%.0f%%"),
            "amount": st.column_config.NumberColumn("amount", format="₹%.0f"),
        },
        disabled=["date", "merchant", "amount", "confidence"],
        hide_index=True,
        width='stretch',
        key="editor",
    )

    changed_mask = edited_df["category"] != display_df["category"]
    n_changed = int(changed_mask.sum())

    col_a, col_b = st.columns([1, 3])
    with col_a:
        retrain_clicked = st.button(
            f"↻ Apply corrections & retrain ({n_changed})",
            disabled=n_changed == 0,
        )
    if retrain_clicked and n_changed > 0:
        corrections = list(zip(
            raw_df.loc[changed_mask, "merchant"].astype(str),
            edited_df.loc[changed_mask, "category"],
        ))
        categorizer.retrain(corrections)
        st.success(
            f"Retrained on {n_changed} correction(s). "
            f"Cross-val accuracy is now {categorizer.cv_accuracy_mean:.0%} "
            f"(was computed over {categorizer.correction_count} total corrections)."
        )
        st.rerun()

    st.write("")
    fig_conf = px.histogram(
        raw_df, x="confidence", nbins=20,
        color_discrete_sequence=["#3987e5"],
        title="Model Confidence Distribution",
    )
    fig_conf.update_layout(yaxis_title="transactions", xaxis_title="confidence")
    st.plotly_chart(themed(fig_conf), width='stretch')

# ---------------------------------------------------------------- Forecast --
with tab_forecast:
    result = predict_next_month(raw_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card_html(
            "Ensemble Prediction", f"₹{result.predicted_amount:,.0f}",
            "average of top-down and bottom-up models",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html(
            "95% Confidence Band", f"₹{result.ci_lower:,.0f} – ₹{result.ci_upper:,.0f}",
            "based on historical residual spread",
        ), unsafe_allow_html=True)
    with c3:
        r2_display = f"{result.r2:.2f}" if result.r2 is not None else "n/a (needs 3+ months)"
        st.markdown(kpi_card_html(
            "Trend Fit (R²)", r2_display, "top-down regression fit quality",
        ), unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        methods = ["Top-down\nregression", "Bottom-up\nregression", "Ensemble\n(final)"]
        values = [result.topdown_amount, result.bottomup_amount, result.predicted_amount]
        colors = [MUTED_INK, MUTED_INK, "#3987e5"]
        fig_methods = go.Figure(go.Bar(x=methods, y=values, marker_color=colors))
        fig_methods.update_layout(title="Forecast Method Comparison", yaxis_title="₹",
                                   showlegend=False)
        st.plotly_chart(themed(fig_methods), width='stretch')
    with col2:
        cat_fc = result.category_forecast.sort_values()
        fig_cat_fc = go.Figure(go.Bar(
            x=cat_fc.values, y=cat_fc.index, orientation="h",
            marker_color=[CATEGORY_COLORS.get(c, MUTED_INK) for c in cat_fc.index],
        ))
        fig_cat_fc.update_layout(title="Next-Month Forecast by Category", xaxis_title="₹",
                                  showlegend=False)
        st.plotly_chart(themed(fig_cat_fc), width='stretch')

    st.caption(
        f"Method: {result.method}. The top-down model fits a trend line directly on monthly "
        "totals; the bottom-up model fits a separate trend per category and sums the results. "
        "Averaging both reduces the risk of either model's blind spots dominating the forecast."
    )

# --------------------------------------------------------------- Anomalies --
with tab_anomalies:
    st.subheader("Unusual Spending (IsolationForest)")
    st.caption(
        "An IsolationForest model scores every transaction on amount, log-amount, and "
        "deviation from its category's average — no fixed threshold rule."
    )
    unusual = detect_unusual(raw_df, contamination=contamination)

    if unusual.empty:
        st.markdown(banner_html("ok", "✅ No unusual transactions detected at this sensitivity."),
                    unsafe_allow_html=True)
    else:
        st.markdown(banner_html("warning",
            f"⚠️ {len(unusual)} transaction(s) flagged as anomalous."), unsafe_allow_html=True)
        unusual_display = unusual[
            ["date", "merchant", "amount", "category", "anomaly_score", "reason"]
        ].copy()
        unusual_display["date"] = unusual_display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            unusual_display,
            width='stretch', hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("amount", format="₹%.0f"),
                "anomaly_score": st.column_config.NumberColumn("anomaly_score", format="%.3f"),
            },
        )

    st.write("")
    plot_df = raw_df.copy()
    plot_df["flag"] = "Normal"
    if not unusual.empty:
        plot_df.loc[unusual.index, "flag"] = "Anomalous"

    fig_scatter = px.scatter(
        plot_df, x="date", y="amount", color="flag",
        color_discrete_map={"Normal": MUTED_INK, "Anomalous": STATUS_COLORS["critical"]},
        hover_data=["merchant", "category"],
        title="Transactions Over Time (flagged in red)",
    )
    fig_scatter.update_traces(marker=dict(size=9))
    st.plotly_chart(themed(fig_scatter), width='stretch')

# ----------------------------------------------------------- Model Insights --
with tab_model:
    st.subheader("Classifier Performance")
    if categorizer.cv_accuracy_mean is None:
        st.info("Not enough examples per category yet for cross-validation.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(kpi_card_html(
                "Cross-Val Accuracy", f"{categorizer.cv_accuracy_mean:.0%}",
                f"± {categorizer.cv_accuracy_std:.0%} · vs. {100 / len(CATEGORIES):.0f}% random baseline",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card_html(
                "Training Examples", str(categorizer.training_example_count),
                f"including {categorizer.correction_count} user correction(s)",
            ), unsafe_allow_html=True)

        st.caption(
            "Cross-validation holds out entire brand names, so this measures the model's "
            "ability to generalize to *unseen* merchants — a genuinely hard one-shot "
            "classification problem — rather than accuracy on merchants it has memorized."
        )

        st.write("")
        col1, col2 = st.columns([1, 1])
        with col1:
            report = categorizer.classification_report_
            rows = []
            for label in categorizer.cv_labels_:
                m = report[label]
                rows.append({
                    "category": label,
                    "precision": round(m["precision"], 2),
                    "recall": round(m["recall"], 2),
                    "f1-score": round(m["f1-score"], 2),
                    "support": int(m["support"]),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        with col2:
            cm = categorizer.confusion_matrix_
            fig_cm = px.imshow(
                cm, x=categorizer.cv_labels_, y=categorizer.cv_labels_,
                color_continuous_scale=SEQUENTIAL_BLUES, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="count"),
                title="Confusion Matrix (cross-validated)",
            )
            fig_cm.update_xaxes(tickangle=45)
            st.plotly_chart(themed(fig_cm), width='stretch')

    st.markdown("---")
    st.subheader("What the Model Looks For")
    chosen_category = st.selectbox("Category", CATEGORIES)
    tokens = categorizer.top_tokens(chosen_category, top_n=10)
    if tokens:
        tokens_df = pd.DataFrame(tokens, columns=["token", "weight"]).sort_values("weight")
        fig_tokens = go.Figure(go.Bar(
            x=tokens_df["weight"], y=tokens_df["token"], orientation="h",
            marker_color=CATEGORY_COLORS.get(chosen_category, MUTED_INK),
        ))
        fig_tokens.update_layout(
            title=f"Top Logistic Regression Weights — {chosen_category}",
            xaxis_title="coefficient weight", showlegend=False,
        )
        st.plotly_chart(themed(fig_tokens), width='stretch')
    else:
        st.caption("No positive-weight tokens found for this category yet.")

with st.expander("Download categorized data"):
    import io
    csv_buffer = io.StringIO()
    raw_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download CSV", csv_buffer.getvalue(),
        file_name="categorized_transactions.csv", mime="text/csv",
    )
