# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.5",
#     "pandas==3.0.2",
#     "plotly==6.7.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.read_csv("P53 Predictions.csv")

    axis_cols = ["Residue Depth", "CA Depth", "RSA"]

    ddg_cols = {
        "Experimental ΔΔG": "DDG",
        "Linear ΔΔG": "Linear",
        "LASSO ΔΔG": "LASSO",
        "Ridge ΔΔG": "Ridge",
        "Random Forest ΔΔG": "RandomForest",
    }

    required = axis_cols + list(ddg_cols.values())
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_df = df.dropna(subset=axis_cols).copy()

    all_ddg = plot_df[list(ddg_cols.values())].to_numpy().ravel()
    finite_ddg = all_ddg[pd.notna(all_ddg)]
    color_abs_max = float(max(abs(finite_ddg.min()), abs(finite_ddg.max())))

    colorscale = [
        [0.0, "#b2182b"],  # negative = red
        [0.5, "#f7f7f7"],  # zero = white
        [1.0, "#1a9850"],  # positive = green
    ]


    def text_col(name):
        if name in plot_df.columns:
            return plot_df[name].astype(str)
        return pd.Series("", index=plot_df.index)


    base_hover = (
        "ID: "
        + text_col("ID")
        + "<br>Mutation: "
        + text_col("Mutation")
        + "<br>PDB: "
        + text_col("PDB")
        + "<br>Residue Depth: "
        + plot_df["Residue Depth"].round(3).astype(str)
        + "<br>CA Depth: "
        + plot_df["CA Depth"].round(3).astype(str)
        + "<br>RSA: "
        + plot_df["RSA"].round(3).astype(str)
    )

    traces = []

    for i, (label, col) in enumerate(ddg_cols.items()):
        model_df = plot_df.dropna(subset=[col])

        traces.append(
            go.Scatter3d(
                x=model_df["Residue Depth"],
                y=model_df["CA Depth"],
                z=model_df["RSA"],
                mode="markers",
                name=label,
                visible=(i == 0),
                marker=dict(
                    size=6,
                    color=model_df[col],
                    colorscale=colorscale,
                    cmin=-color_abs_max,
                    cmax=color_abs_max,
                    cmid=0,
                    opacity=0.9,
                    colorbar=dict(title="ΔΔG"),
                    line=dict(width=0.4, color="rgba(0,0,0,0.35)"),
                ),
                text=(
                    base_hover.loc[model_df.index]
                    + f"<br>{label}: "
                    + model_df[col].round(3).astype(str)
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    buttons = []

    for i, label in enumerate(ddg_cols.keys()):
        visible = [False] * len(ddg_cols)
        visible[i] = True

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"P53 3D plot colored by {label}"},
                ],
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title="P53 3D plot colored by Experimental ΔΔG",
        width=950,
        height=720,
        template="plotly_white",
        scene=dict(
            xaxis_title="Residue Depth",
            yaxis_title="CA Depth",
            zaxis_title="RSA",
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig
    return


if __name__ == "__main__":
    app.run()
