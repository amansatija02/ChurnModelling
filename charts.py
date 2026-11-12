"""
charts.py  --  all Plotly charts in one place so app.py stays short.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from churn_utils import LABELS, NUMERIC, TARGET, churn_by_segment, segment_series

RED, TEAL, AMBER, NAVY, GREY = "#E4572E", "#2A9D8F", "#E9A23B", "#264653", "#9AA5B1"
MODEL_COLORS = [NAVY, TEAL, AMBER, RED]


def style(fig, height=360):
    """Same clean look for every chart."""
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=-0.2))
    return fig


def show_value(value):
    """Text for hover: round numbers, keep text (like country) as it is."""
    if isinstance(value, str):
        return value
    return f"{value:,.0f}"


# ------------------------------------------------------------ prediction charts
def gauge(prob, threshold):
    """Speedometer for one churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number=dict(suffix="%", font=dict(size=44)),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=NAVY, thickness=0.25),
            steps=[dict(range=[0, threshold * 50], color="#CDEBE7"),
                   dict(range=[threshold * 50, threshold * 100], color="#FBE3B5"),
                   dict(range=[threshold * 100, 100], color="#F7C8BA")],
            threshold=dict(line=dict(color=RED, width=4), value=threshold * 100),
        ),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0))
    return fig


def reasons_bar(table):
    """Red bars push the customer towards leaving, teal bars keep them loyal."""
    t = table.sort_values("impact")
    colors = [RED if v > 0 else TEAL for v in t["impact"]]
    hover = [f"This customer: {show_value(a)}<br>Typical loyal customer: {show_value(b)}"
             for a, b in zip(t["your_value"], t["typical_value"])]
    fig = go.Figure(go.Bar(
        x=t["impact"], y=t["name"], orientation="h", marker_color=colors,
        text=[f"{v:+.1f} pts" for v in t["impact"]], textposition="outside",
        hovertext=hover, hoverinfo="text+y",
    ))
    fig.add_vline(x=0, line_color=GREY)
    fig.update_layout(title="Why? What drives this customer's churn risk",
                      xaxis_title="Effect on churn probability (percentage points)")
    fig.update_xaxes(range=[min(0, t["impact"].min()) - 10, t["impact"].max() + 12])
    return style(fig, 400)


# ------------------------------------------------------------ insight charts
def segment_bar(df, feature):
    """Churn rate for each group of one feature."""
    t = churn_by_segment(df, feature)
    overall = df[TARGET].mean()
    colors = [RED if r > overall else TEAL for r in t["churn_rate"]]
    labels = [f"{r:.0%}<br>n={n:,}" for r, n in zip(t["churn_rate"], t["customers"])]
    fig = go.Figure(go.Bar(x=t["segment"], y=t["churn_rate"] * 100, marker_color=colors,
                           text=labels, textposition="outside"))
    fig.add_hline(y=overall * 100, line_dash="dash", line_color=GREY)
    fig.update_layout(title=f"Churn rate by {LABELS[feature].lower()}  (dashed line = average {overall:.0%})",
                      yaxis_title="Churn rate (%)")
    fig.update_yaxes(range=[0, 120], tickvals=[0, 25, 50, 75, 100])
    return style(fig)


def heatmap(df, row_feature, col_feature):
    """Churn rate for combinations of two features."""
    data = pd.DataFrame({
        "row": segment_series(df, row_feature).astype(str),
        "col": segment_series(df, col_feature).astype(str),
        "churn": df[TARGET],
    })
    rate = data.pivot_table(index="row", columns="col", values="churn", aggfunc="mean")
    count = data.pivot_table(index="row", columns="col", values="churn", aggfunc="size")
    rate = rate.where(count >= 30) * 100                  # hide groups with very few customers

    text = []
    for r in rate.values:
        text.append([f"{v:.0f}%" if v == v else "" for v in r])   # v == v is False for NaN

    fig = go.Figure(go.Heatmap(
        z=rate.values, x=rate.columns, y=rate.index, text=text, texttemplate="%{text}",
        colorscale=[[0, "#E7F4F2"], [0.5, "#F6C9A8"], [1, RED]], colorbar=dict(title="Churn %"),
    ))
    fig.update_layout(title=f"Churn rate: {LABELS[row_feature]} vs {LABELS[col_feature]} "
                            "(groups under 30 customers hidden)")
    return style(fig, 420)


def distribution(df, feature):
    """Stayed vs left for a numeric feature."""
    plot_df = df.assign(Status=df[TARGET].map({0: "Stayed", 1: "Left"}))
    fig = px.histogram(plot_df, x=feature, color="Status", barmode="overlay",
                       histnorm="percent", opacity=0.65, nbins=30,
                       color_discrete_map={"Stayed": TEAL, "Left": RED},
                       labels={feature: LABELS[feature]})
    fig.update_layout(title=f"{LABELS[feature]}: customers who stayed vs left",
                      yaxis_title="% of group")
    return style(fig)


def correlation_heatmap(df):
    corr = df[NUMERIC + ["HasCrCard", "IsActiveMember", TARGET]].corr().round(2)
    names = [LABELS.get(c, "Churned") for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1,
                               colorscale="RdBu_r", text=corr.values, texttemplate="%{text}"))
    fig.update_layout(title="Correlation between numeric features")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 460)


def importance_bar(importance, model_name):
    """Which features matter most to one model."""
    t = importance.sort_values("importance")
    fig = go.Figure(go.Bar(x=t["importance"], y=t["name"], orientation="h", marker_color=NAVY,
                           text=[f"{v:.3f}" for v in t["importance"]], textposition="outside"))
    fig.update_layout(title=f"What drives churn overall? ({model_name})",
                      xaxis_title="Drop in AUC when the feature is shuffled")
    fig.update_xaxes(range=[0, t["importance"].max() * 1.2])
    return style(fig, 400)


# ------------------------------------------------------------ performance charts
def roc_curve_fig(roc, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"Model (AUC {auc:.3f})",
                             line=dict(color=NAVY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=GREY, dash="dash")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return style(fig, 380)


def confusion_fig(matrix):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Predicted Stayed", "Predicted Left"], y=["Actually Stayed", "Actually Left"],
        colorscale=[[0, "#F1F5F9"], [1, NAVY]], text=matrix, texttemplate="%{text}",
        textfont=dict(size=18), showscale=False))
    fig.update_layout(title="Confusion matrix (test data)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 380)


# ------------------------------------------------------------ comparison charts (option 3)
def compare_metrics_bar(results):
    """Grouped bars: each metric, one bar per model."""
    fig = go.Figure()
    for i, name in enumerate(results):
        s = results[name]["scores"]
        fig.add_trace(go.Bar(
            name=name, x=["ROC-AUC", "Precision", "Recall", "F1"],
            y=[s["roc_auc"], s["precision"], s["recall"], s["f1"]],
            marker_color=MODEL_COLORS[i], text=[f"{v:.2f}" for v in
                                                [s["roc_auc"], s["precision"], s["recall"], s["f1"]]],
            textposition="outside"))
    fig.update_layout(title="Model comparison on test data", barmode="group")
    fig.update_yaxes(range=[0, 1.1])
    return style(fig, 420)


def compare_roc(results):
    fig = go.Figure()
    for i, name in enumerate(results):
        roc = results[name]["roc"]
        auc = results[name]["scores"]["roc_auc"]
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"{name} ({auc:.3f})",
                                 line=dict(color=MODEL_COLORS[i], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=GREY, dash="dash")))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return style(fig, 420)


def compare_importance(importance):
    """Do all models agree on what drives churn?"""
    fig = go.Figure()
    for i, name in enumerate(importance):
        t = importance[name].sort_values("importance")
        fig.add_trace(go.Bar(name=name, x=t["importance"], y=t["name"], orientation="h",
                             marker_color=MODEL_COLORS[i]))
    fig.update_layout(title="Feature importance by model", barmode="group",
                      xaxis_title="Drop in AUC when the feature is shuffled")
    return style(fig, 600)


def compare_probabilities(names, probs, thresholds):
    """Churn probability for the same customer from each model."""
    colors = [RED if p >= thresholds[n] else TEAL for n, p in zip(names, probs)]
    fig = go.Figure(go.Bar(x=names, y=[p * 100 for p in probs], marker_color=colors,
                           text=[f"{p:.0%}" for p in probs], textposition="outside"))
    fig.update_layout(title="Churn probability for this customer (red = predicted to leave)",
                      yaxis_title="Churn probability (%)")
    fig.update_yaxes(range=[0, 115])
    return style(fig, 360)
