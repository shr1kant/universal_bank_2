import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank – Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp { background-color: #0f1117; color: #e0e0e0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2535 0%, #252d40 100%);
    border: 1px solid #2e3a50;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card h2 { color: #4fc3f7; font-size: 2rem; margin: 0; }
.metric-card p  { color: #90a0b0; font-size: 0.85rem; margin: 4px 0 0; }
.metric-card .delta { color: #66bb6a; font-size: 0.8rem; }

/* Section headers */
.section-header {
    color: #4fc3f7;
    font-size: 1.4rem;
    font-weight: 700;
    border-left: 4px solid #4fc3f7;
    padding-left: 12px;
    margin: 24px 0 12px;
}

/* Tab style override */
[data-testid="stTab"] button {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1b2838 0%, #1e2d40 100%);
    border-left: 4px solid #42a5f5;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    line-height: 1.6;
}
.insight-box strong { color: #81d4fa; }

/* Prescriptive cards */
.rec-card {
    background: linear-gradient(135deg, #1a2820 0%, #1e3025 100%);
    border: 1px solid #2e5040;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    border-left: 4px solid #66bb6a;
}
.rec-card h4 { color: #a5d6a7; margin: 0 0 8px; }
.rec-card p  { color: #c8d8c8; font-size: 0.88rem; margin: 0; }

.warn-card {
    background: linear-gradient(135deg, #281a1a 0%, #301e1e 100%);
    border-left: 4px solid #ef5350;
}
.warn-card h4 { color: #ef9a9a; }
.warn-card p  { color: #d8c8c8; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

@st.cache_data
def load_data():
    csv_path = _os.path.join(_HERE, "UniversalBank.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["ID", "ZIP Code"], errors="ignore")
    df["Education_Label"] = df["Education"].map(
        {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}
    )
    df["Income_Group"] = pd.cut(
        df["Income"],
        bins=[0, 50, 100, 150, 200, 300],
        labels=["<50K", "50-100K", "100-150K", "150-200K", ">200K"],
    )
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"],
    )
    df["Loan_Status"] = df["Personal Loan"].map({0: "Rejected", 1: "Accepted"})
    return df

df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("### Dashboard Filters")
    st.markdown("---")

    income_min, income_max = int(df_full["Income"].min()), int(df_full["Income"].max())
    income_range = st.slider(
        "💰 Income Range (K$)", income_min, income_max, (income_min, income_max)
    )

    edu_options = ["All"] + list(df_full["Education_Label"].unique())
    edu_filter = st.multiselect("🎓 Education Level", edu_options[1:], default=edu_options[1:])

    family_options = sorted(df_full["Family"].unique())
    family_filter = st.multiselect("👨‍👩‍👧 Family Size", family_options, default=family_options)

    age_groups = list(df_full["Age_Group"].cat.categories)
    age_filter = st.multiselect("🎂 Age Group", age_groups, default=age_groups)

    st.markdown("---")
    st.markdown("**🎯 Objective**")
    st.markdown(
        "_Identify customers most likely to accept a Personal Loan offer._",
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.caption("Dataset: 5,000 bank customers | 14 features")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full[
    (df_full["Income"] >= income_range[0])
    & (df_full["Income"] <= income_range[1])
    & (df_full["Education_Label"].isin(edu_filter))
    & (df_full["Family"].isin(family_filter))
    & (df_full["Age_Group"].isin(age_filter))
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#4fc3f7;margin-bottom:0'>🏦 Universal Bank – Personal Loan Analytics</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#78909c;margin-top:4px'>"
    "A 360° analytics dashboard: Descriptive · Diagnostic · Predictive · Prescriptive</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
total      = len(df)
accepted   = int(df["Personal Loan"].sum())
accept_pct = accepted / total * 100 if total else 0
avg_income = df["Income"].mean()
avg_cc     = df["CCAvg"].mean()
avg_mort   = df["Mortgage"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "👥 Total Customers", f"{total:,}", "Filtered dataset"),
    (c2, "✅ Loan Accepted",   f"{accepted:,}", f"{accept_pct:.1f}% acceptance rate"),
    (c3, "💰 Avg Income",     f"${avg_income:.0f}K", "Annual income"),
    (c4, "💳 Avg CC Spend",   f"${avg_cc:.2f}K/mo", "Monthly CC average"),
    (c5, "🏠 Avg Mortgage",   f"${avg_mort:.0f}K", "Mortgage balance"),
]
for col, title, val, sub in kpis:
    with col:
        st.markdown(
            f"<div class='metric-card'><h2>{val}</h2><p>{title}</p>"
            f"<p class='delta'>{sub}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Descriptive",
    "🔍 Diagnostic",
    "🤖 Predictive",
    "🎯 Prescriptive",
    "📈 Interactive Drill-Down",
])

DARK_BG   = "#0f1117"
CARD_BG   = "#1e2535"
ACCENT    = "#4fc3f7"
GREEN     = "#66bb6a"
RED_COLOR = "#ef5350"
ORANGE    = "#ffa726"
PALETTE   = [ACCENT, GREEN, ORANGE, RED_COLOR, "#ab47bc", "#26c6da", "#ec407a"]

def dark_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=15)),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#c0c8d8"),
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(bgcolor=CARD_BG, bordercolor="#2e3a50", borderwidth=1),
        xaxis=dict(gridcolor="#2a3040", zerolinecolor="#2a3040"),
        yaxis=dict(gridcolor="#2a3040", zerolinecolor="#2a3040"),
    )
    return fig


# ══════════════════════════════════════════════════════
# TAB 1 – DESCRIPTIVE
# ══════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>Descriptive Analytics — Who Are Our Customers?</div>", unsafe_allow_html=True)

    # ── Row 1: Loan acceptance donut + Education bar ──
    col1, col2 = st.columns(2)

    with col1:
        counts = df["Loan_Status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.55,
            marker=dict(colors=[GREEN, RED_COLOR]),
            textinfo="percent+label",
            textfont=dict(size=13),
        ))
        fig.add_annotation(
            text=f"<b>{accept_pct:.1f}%</b><br>Accepted",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e0e0e0"),
        )
        dark_layout(fig, "🎯 Personal Loan Acceptance", 380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>💡 Only <strong>9.6%</strong> of customers accepted a personal loan. "
            "This class imbalance is important for modeling — the bank has a large untapped pool to target.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        edu_loan = (
            df.groupby("Education_Label")["Personal Loan"]
            .agg(["sum", "count"])
            .reset_index()
        )
        edu_loan["Rate"] = edu_loan["sum"] / edu_loan["count"] * 100
        fig = px.bar(
            edu_loan, x="Education_Label", y="Rate",
            color="Education_Label",
            color_discrete_sequence=PALETTE,
            text=edu_loan["Rate"].apply(lambda x: f"{x:.1f}%"),
            labels={"Education_Label": "Education", "Rate": "Acceptance Rate (%)"},
        )
        fig.update_traces(textposition="outside")
        dark_layout(fig, "🎓 Loan Acceptance Rate by Education Level", 380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>💡 <strong>Advanced/Professional degree</strong> holders show the highest "
            "loan acceptance rate. Higher education correlates strongly with loan uptake.</div>",
            unsafe_allow_html=True,
        )

    # ── Row 2: Age & Income distributions ──
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df, x="Age", color="Loan_Status",
            barmode="overlay", nbins=30,
            color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
            opacity=0.75,
            labels={"Age": "Age (years)", "count": "Number of Customers"},
        )
        dark_layout(fig, "🎂 Age Distribution by Loan Status", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df, x="Income", color="Loan_Status",
            barmode="overlay", nbins=40,
            color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
            opacity=0.75,
            labels={"Income": "Annual Income ($K)", "count": "Number of Customers"},
        )
        dark_layout(fig, "💰 Income Distribution by Loan Status", 360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Family size + CCAvg ──
    col5, col6 = st.columns(2)

    with col5:
        fam = df.groupby("Family")["Personal Loan"].agg(["sum", "count"]).reset_index()
        fam["Rate"] = fam["sum"] / fam["count"] * 100
        fig = px.bar(
            fam, x="Family", y="Rate",
            color="Family",
            color_discrete_sequence=PALETTE,
            text=fam["Rate"].apply(lambda x: f"{x:.1f}%"),
            labels={"Family": "Family Size", "Rate": "Acceptance Rate (%)"},
        )
        fig.update_traces(textposition="outside")
        dark_layout(fig, "👨‍👩‍👧 Loan Acceptance by Family Size", 360)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = px.box(
            df, x="Loan_Status", y="CCAvg",
            color="Loan_Status",
            color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
            labels={"CCAvg": "Monthly CC Spend ($K)", "Loan_Status": "Loan Status"},
        )
        dark_layout(fig, "💳 Credit Card Spending Distribution", 360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats table ──
    st.markdown("<div class='section-header'>📋 Summary Statistics</div>", unsafe_allow_html=True)
    num_cols = ["Age", "Income", "CCAvg", "Mortgage", "Experience"]
    summary = df[num_cols].describe().round(2)
    st.dataframe(summary, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2 – DIAGNOSTIC
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>Diagnostic Analytics — What Drives Loan Acceptance?</div>", unsafe_allow_html=True)

    accepted_df = df[df["Personal Loan"] == 1]
    rejected_df = df[df["Personal Loan"] == 0]

    # ── Row 1: Mean comparison grouped bar ──
    compare_cols = ["Income", "CCAvg", "Mortgage", "Age", "Experience"]
    compare_data = pd.DataFrame({
        "Feature":  compare_cols * 2,
        "Group":    ["Accepted"] * len(compare_cols) + ["Rejected"] * len(compare_cols),
        "Mean":     [accepted_df[c].mean() for c in compare_cols] +
                    [rejected_df[c].mean() for c in compare_cols],
    })

    fig = px.bar(
        compare_data, x="Feature", y="Mean", color="Group", barmode="group",
        color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
        text=compare_data["Mean"].round(1).astype(str),
    )
    fig.update_traces(textposition="outside")
    dark_layout(fig, "📊 Key Feature Means: Accepted vs Rejected Customers", 400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>💡 Customers who accepted loans have <strong>significantly higher income "
        "and credit card spend</strong>. Income (~$144K vs $66K) and CCAvg (~$3.9K vs $1.7K/month) "
        "are the strongest differentiators.</div>",
        unsafe_allow_html=True,
    )

    # ── Row 2: Income vs CC scatter + Violin ──
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df.sample(min(2000, len(df))), x="Income", y="CCAvg",
            color="Loan_Status",
            color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
            opacity=0.6, size_max=6,
            labels={"Income": "Annual Income ($K)", "CCAvg": "Monthly CC Spend ($K)"},
            hover_data=["Education_Label", "Family"],
        )
        dark_layout(fig, "💰 Income vs Credit Card Spend", 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(
            df, x="Loan_Status", y="Income",
            color="Loan_Status",
            color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a"},
            box=True, points="outliers",
            labels={"Income": "Annual Income ($K)", "Loan_Status": "Loan Status"},
        )
        dark_layout(fig, "🎻 Income Distribution Violin Plot", 400)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Banking services analysis ──
    st.markdown("<div class='section-header'>🏛️ Banking Services & Loan Acceptance</div>", unsafe_allow_html=True)

    services = ["Securities Account", "CD Account", "Online", "CreditCard"]
    service_rates = []
    for s in services:
        for val in [0, 1]:
            sub = df[df[s] == val]
            rate = sub["Personal Loan"].mean() * 100
            service_rates.append({
                "Service": s, "Has Service": "Yes" if val == 1 else "No",
                "Acceptance Rate (%)": rate, "Count": len(sub),
            })
    svc_df = pd.DataFrame(service_rates)

    fig = px.bar(
        svc_df, x="Service", y="Acceptance Rate (%)", color="Has Service",
        barmode="group",
        color_discrete_map={"Yes": ACCENT, "No": "#546e7a"},
        text=svc_df["Acceptance Rate (%)"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    dark_layout(fig, "🏦 Loan Acceptance Rate by Banking Service Usage", 400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>💡 Customers with a <strong>CD Account</strong> have dramatically "
        "higher loan acceptance (~20% vs ~8%). <strong>Securities Account</strong> holders also show "
        "higher rates. Online banking and credit card usage have a smaller but measurable effect.</div>",
        unsafe_allow_html=True,
    )

    # ── Correlation heatmap ──
    st.markdown("<div class='section-header'>🔗 Correlation Heatmap</div>", unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Personal Loan"], errors="ignore")
    corr = df.select_dtypes(include=[np.number]).corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
    ))
    dark_layout(fig, "Correlation Matrix — All Features", 520)
    fig.update_layout(margin=dict(l=80, r=30, t=60, b=80))
    st.plotly_chart(fig, use_container_width=True)

    # ── Key driver insights ──
    st.markdown("<div class='section-header'>🔑 Key Drivers Ranked</div>", unsafe_allow_html=True)
    corr_target = (
        df.select_dtypes(include=[np.number])
        .corr()["Personal Loan"]
        .drop("Personal Loan")
        .abs()
        .sort_values(ascending=True)
    )
    fig = px.bar(
        x=corr_target.values, y=corr_target.index,
        orientation="h",
        color=corr_target.values,
        color_continuous_scale="Blues",
        labels={"x": "Absolute Correlation with Personal Loan", "y": "Feature"},
    )
    dark_layout(fig, "Feature Correlation with Personal Loan (Absolute)", 400)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 – PREDICTIVE
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>Predictive Analytics — Who Will Accept a Loan?</div>", unsafe_allow_html=True)

    features = ["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"]

    @st.cache_data
    def train_models(data):
        X = data[features].copy()
        y = data["Personal Loan"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        results = {}
        for name, model in models.items():
            Xtr = X_train_sc if name == "Logistic Regression" else X_train
            Xte = X_test_sc  if name == "Logistic Regression" else X_test
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:, 1]
            results[name] = {
                "model":    model,
                "X_test":   Xte,
                "y_test":   y_test,
                "y_pred":   y_pred,
                "y_prob":   y_prob,
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc":  roc_auc_score(y_test, y_prob),
                "report":   classification_report(y_test, y_pred, output_dict=True),
            }
        return results, X_train, X_test, y_train, y_test, scaler

    with st.spinner("🤖 Training models…"):
        results, X_train, X_test, y_train, y_test, scaler = train_models(df_full)

    # ── Model comparison ──
    model_summary = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"], "ROC-AUC": v["roc_auc"]}
        for k, v in results.items()
    ]).sort_values("ROC-AUC", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            model_summary, x="Model", y="ROC-AUC",
            color="ROC-AUC", color_continuous_scale="Blues",
            text=model_summary["ROC-AUC"].apply(lambda x: f"{x:.3f}"),
        )
        fig.update_traces(textposition="outside")
        dark_layout(fig, "🏆 Model ROC-AUC Comparison", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        colors_roc = [ACCENT, GREEN, ORANGE, "#ab47bc"]
        for (name, res), col in zip(results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=col, width=2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#546e7a"), name="Random",
        ))
        dark_layout(fig, "📈 ROC Curves – All Models", 380)
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Best model detail ──
    best_name = model_summary.iloc[0]["Model"]
    best = results[best_name]
    st.markdown(f"<div class='section-header'>🥇 Best Model: {best_name}</div>", unsafe_allow_html=True)

    cm = confusion_matrix(best["y_test"], best["y_pred"])
    col3, col4 = st.columns(2)

    with col3:
        fig = px.imshow(
            cm, text_auto=True,
            x=["Predicted: Rejected", "Predicted: Accepted"],
            y=["Actual: Rejected", "Actual: Accepted"],
            color_continuous_scale="Blues",
        )
        dark_layout(fig, f"Confusion Matrix – {best_name}", 350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Feature importance
        if hasattr(best["model"], "feature_importances_"):
            fi = pd.Series(best["model"].feature_importances_, index=features).sort_values(ascending=True)
        else:
            fi = pd.Series(np.abs(best["model"].coef_[0]), index=features).sort_values(ascending=True)

        fig = px.bar(
            x=fi.values, y=fi.index, orientation="h",
            color=fi.values, color_continuous_scale="Blues",
            labels={"x": "Importance", "y": "Feature"},
        )
        dark_layout(fig, f"Feature Importance – {best_name}", 350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Interactive predictor ──
    st.markdown("<div class='section-header'>🔮 Live Loan Acceptance Predictor</div>", unsafe_allow_html=True)
    st.markdown("Adjust the sliders below to predict whether a customer will accept a personal loan.")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_age      = st.slider("Age",        18, 70, 35)
        p_exp      = st.slider("Experience", 0,  45, 10)
        p_income   = st.slider("Income ($K)", 8, 224, 80)
        p_family   = st.slider("Family Size", 1, 4, 2)
    with pc2:
        p_ccavg    = st.slider("Monthly CC Spend ($K)", 0.0, 10.0, 2.0, 0.1)
        p_edu      = st.selectbox("Education", [1, 2, 3],
                                  format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
        p_mortgage = st.slider("Mortgage ($K)", 0, 635, 100)
    with pc3:
        p_sec      = st.checkbox("Securities Account")
        p_cd       = st.checkbox("CD Account")
        p_online   = st.checkbox("Online Banking")
        p_cc       = st.checkbox("Credit Card")

    input_data = pd.DataFrame([[p_age, p_exp, p_income, p_family, p_ccavg,
                                  p_edu, p_mortgage,
                                  int(p_sec), int(p_cd), int(p_online), int(p_cc)]],
                               columns=features)

    rf_model = results["Random Forest"]["model"]
    prob = rf_model.predict_proba(input_data)[0][1]
    pred = "✅ LIKELY TO ACCEPT" if prob >= 0.5 else "❌ UNLIKELY TO ACCEPT"
    color = GREEN if prob >= 0.5 else RED_COLOR

    st.markdown(
        f"<div style='background:{CARD_BG};border:2px solid {color};border-radius:12px;"
        f"padding:20px;text-align:center;margin-top:12px'>"
        f"<h2 style='color:{color}'>{pred}</h2>"
        f"<p style='color:#90a0b0'>Probability of Acceptance: "
        f"<strong style='color:{color}'>{prob*100:.1f}%</strong></p></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════
# TAB 4 – PRESCRIPTIVE
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>Prescriptive Analytics — How Should the Bank Act?</div>", unsafe_allow_html=True)

    # Acceptance by income group + education heatmap
    col1, col2 = st.columns(2)

    with col1:
        ig = df_full.groupby("Income_Group")["Personal Loan"].agg(["sum","count"]).reset_index()
        ig["Rate"] = ig["sum"] / ig["count"] * 100
        fig = px.bar(
            ig, x="Income_Group", y="Rate",
            color="Rate", color_continuous_scale="Blues",
            text=ig["Rate"].apply(lambda x: f"{x:.1f}%"),
            labels={"Income_Group": "Income Group", "Rate": "Acceptance Rate (%)"},
        )
        fig.update_traces(textposition="outside")
        dark_layout(fig, "💰 Loan Acceptance Rate by Income Segment", 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        heat_data = df_full.groupby(["Education_Label", "Income_Group"])["Personal Loan"].mean().unstack(fill_value=0) * 100
        fig = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns.astype(str).tolist(),
            y=heat_data.index.tolist(),
            colorscale="Blues",
            text=np.round(heat_data.values, 1).astype(str),
            texttemplate="%{text}%",
            textfont=dict(size=11),
        ))
        dark_layout(fig, "🎓 Acceptance Rate: Education × Income Group", 380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Segment profiler ──
    st.markdown("<div class='section-header'>🎯 High-Value Segment Profiler</div>", unsafe_allow_html=True)

    segments = df_full.groupby(["Education_Label", "Income_Group"]).agg(
        Customers=("Personal Loan", "count"),
        Accepted=("Personal Loan", "sum"),
    ).reset_index()
    segments["Rate"] = segments["Accepted"] / segments["Customers"] * 100
    segments = segments[segments["Customers"] >= 30].sort_values("Rate", ascending=False).head(10)

    fig = px.scatter(
        segments, x="Customers", y="Rate",
        size="Accepted", color="Education_Label",
        hover_data=["Income_Group", "Customers", "Accepted"],
        color_discrete_sequence=PALETTE,
        labels={"Rate": "Acceptance Rate (%)", "Customers": "Segment Size"},
        size_max=40,
    )
    dark_layout(fig, "Segment Bubble Chart: Size = Accepted Loans", 430)
    st.plotly_chart(fig, use_container_width=True)

    # ── Action recommendations ──
    st.markdown("<div class='section-header'>📋 Strategic Recommendations</div>", unsafe_allow_html=True)

    recs = [
        ("🥇 Priority Segment: High-Income Professionals",
         "Target customers earning >$100K with Graduate or Advanced degrees. "
         "This segment shows 30–50%+ acceptance rates. Offer premium, low-rate personal loans with "
         "fast approval and minimal paperwork. Use personalized email and in-app campaigns."),
        ("🥈 CD Account Holders",
         "Customers with a CD Account convert at ~3× the base rate. "
         "These customers already trust the bank with deposits. Offer pre-approved loan bundles "
         "as rewards for loyalty, with special interest rates."),
        ("🥉 Securities Account Holders",
         "These customers are financially engaged and show higher loan interest. "
         "Use cross-sell campaigns timed with securities renewals or market events."),
        ("💡 Family Size 3–4 with Mid Income",
         "Families with 3–4 members and incomes between $50K–$120K show meaningful uptake. "
         "Market education loans, home improvement loans, and family-needs financing to this group."),
        ("📱 Online Banking Users",
         "Online users are slightly more likely to accept. Invest in targeted in-app notifications, "
         "pre-filled digital loan applications, and one-click apply features."),
    ]
    avoid = [
        ("⚠️ Avoid: Low-Income, Low-Education Segment",
         "Customers earning <$50K with undergrad education have <5% acceptance and higher default risk. "
         "Avoid high-cost marketing to this group; instead explore micro-finance or secured products."),
    ]

    for title, text in recs:
        st.markdown(
            f"<div class='rec-card'><h4>{title}</h4><p>{text}</p></div>",
            unsafe_allow_html=True,
        )
    for title, text in avoid:
        st.markdown(
            f"<div class='rec-card warn-card'><h4>{title}</h4><p>{text}</p></div>",
            unsafe_allow_html=True,
        )

    # ── Expected ROI calculator ──
    st.markdown("<div class='section-header'>💹 Campaign ROI Estimator</div>", unsafe_allow_html=True)

    ri1, ri2, ri3 = st.columns(3)
    with ri1:
        campaign_size  = st.number_input("Campaign Audience Size", 100, 50000, 5000, 100)
        target_segment = st.selectbox("Target Segment", ["All Customers (~9.6%)", "High Income + Advanced Degree (~45%)", "CD Account Holders (~20%)", "Custom"])
    with ri2:
        if target_segment == "All Customers (~9.6%)":        rate_est = 0.096
        elif target_segment == "High Income + Advanced Degree (~45%)": rate_est = 0.45
        elif target_segment == "CD Account Holders (~20%)":  rate_est = 0.20
        else: rate_est = st.slider("Custom Acceptance Rate (%)", 1, 100, 15) / 100
        loan_amount  = st.number_input("Avg Loan Amount ($K)", 10, 500, 50, 5)
        margin_pct   = st.slider("Bank Margin (%)", 1, 10, 3)
    with ri3:
        cost_per_contact = st.number_input("Cost per Contact ($)", 1, 100, 5)

    expected_conversions = int(campaign_size * rate_est)
    total_cost           = campaign_size * cost_per_contact
    revenue              = expected_conversions * loan_amount * 1000 * (margin_pct / 100)
    net_roi              = revenue - total_cost

    roi_cols = st.columns(4)
    for col, label, val, clr in zip(
        roi_cols,
        ["Expected Conversions", "Campaign Cost", "Revenue (Margin)", "Net ROI"],
        [f"{expected_conversions:,}", f"${total_cost:,.0f}", f"${revenue:,.0f}", f"${net_roi:,.0f}"],
        [ACCENT, ORANGE, GREEN, GREEN if net_roi > 0 else RED_COLOR],
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><h2 style='color:{clr}'>{val}</h2><p>{label}</p></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════
# TAB 5 – INTERACTIVE DRILL-DOWN
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>📈 Interactive Drill-Down Visualizations</div>", unsafe_allow_html=True)
    st.markdown("Explore multi-level breakdowns interactively. Select a primary dimension to drill into.")

    drill_dim = st.radio(
        "Select primary breakdown dimension:",
        ["Education → Income Group → Loan Status",
         "Income Group → Education → Loan Status",
         "Family Size → Education → Loan Status",
         "Age Group → Income Group → Loan Status"],
        horizontal=True,
    )

    if drill_dim.startswith("Education"):
        path = ["Education_Label", "Income_Group", "Loan_Status"]
    elif drill_dim.startswith("Income Group"):
        path = ["Income_Group", "Education_Label", "Loan_Status"]
    elif drill_dim.startswith("Family"):
        df_full["Family_Label"] = "Family " + df_full["Family"].astype(str)
        path = ["Family_Label", "Education_Label", "Loan_Status"]
    else:
        path = ["Age_Group", "Income_Group", "Loan_Status"]

    agg_df = df_full.copy()
    if "Family_Label" not in agg_df.columns:
        agg_df["Family_Label"] = "Family " + agg_df["Family"].astype(str)
    agg_df["Age_Group"] = agg_df["Age_Group"].astype(str)
    agg_df["Income_Group"] = agg_df["Income_Group"].astype(str)

    fig = px.sunburst(
        agg_df, path=path, values=None,
        color="Loan_Status" if "Loan_Status" in path else path[0],
        color_discrete_map={"Accepted": GREEN, "Rejected": "#546e7a", "(?)": "#2e3a50"},
    )
    fig.update_traces(textinfo="label+percent parent")
    dark_layout(fig, f"☀️ Sunburst: {drill_dim}", 560)
    fig.update_layout(paper_bgcolor=CARD_BG)
    st.plotly_chart(fig, use_container_width=True)

    # ── Treemap ──
    st.markdown("<div class='section-header'>🗺️ Treemap: Volume & Acceptance Rate</div>", unsafe_allow_html=True)

    tree_group = ["Education_Label", "Income_Group"]
    tree_df = df_full.groupby(tree_group).agg(
        Customers=("Personal Loan", "count"),
        Accepted=("Personal Loan", "sum"),
    ).reset_index()
    tree_df["Acceptance Rate"] = (tree_df["Accepted"] / tree_df["Customers"] * 100).round(1)
    tree_df["Label"] = tree_df["Education_Label"] + " | " + tree_df["Income_Group"].astype(str)

    fig = px.treemap(
        tree_df,
        path=["Education_Label", "Income_Group"],
        values="Customers",
        color="Acceptance Rate",
        color_continuous_scale="Blues",
        hover_data=["Customers", "Accepted", "Acceptance Rate"],
    )
    dark_layout(fig, "Treemap: Segment Size (area) & Acceptance Rate (color)", 500)
    fig.update_layout(paper_bgcolor=CARD_BG)
    st.plotly_chart(fig, use_container_width=True)

    # ── Parallel coordinates ──
    st.markdown("<div class='section-header'>🔀 Parallel Coordinates — Multi-Feature Pattern Explorer</div>", unsafe_allow_html=True)

    sample = df_full.sample(min(1500, len(df_full)), random_state=42)
    fig = px.parallel_coordinates(
        sample,
        dimensions=["Age", "Income", "CCAvg", "Mortgage", "Family", "Education"],
        color="Personal Loan",
        color_continuous_scale=[[0, "#546e7a"], [1, GREEN]],
        labels={"Personal Loan": "Loan"},
    )
    dark_layout(fig, "Parallel Coordinates: Customer Profile Patterns", 450)
    fig.update_layout(paper_bgcolor=CARD_BG)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>💡 In the parallel coordinates chart, <strong>green lines = loan accepted</strong>. "
        "You can click-drag on any axis to filter specific ranges and trace customer profiles that convert. "
        "Notice that accepting customers cluster at high Income, high CCAvg, higher Education values.</div>",
        unsafe_allow_html=True,
    )
