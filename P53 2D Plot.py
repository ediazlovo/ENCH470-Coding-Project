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

    x_col = "Residue Depth"
    y_col = "RSA"

    models = {
        "Experimental": "DDG",
        "Linear": "Linear",
        "LASSO": "LASSO",
        "Ridge": "Ridge",
        "Random Forest": "RandomForest",
    }

    required_cols = [x_col, y_col, *models.values()]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    ddg_values = plot_df[list(models.values())].stack()
    color_abs_max = float(ddg_values.abs().max())

    colorscale = [
        [0.00, "#C92589"],
        [0.25, "#E29BC7"],
        [0.50, "#B0B0B0"],
        [0.75, "#A6D672"],
        [1.00, "#3FA323"],
    ]

    def optional_text(col):
        return plot_df[col].astype(str) if col in plot_df.columns else ""

    hover_base = (
        "ID: " + optional_text("ID")
        + "<br>Mutation: " + optional_text("Mutation")
        + "<br>PDB: " + optional_text("PDB")
        + f"<br>{x_col}: " + plot_df[x_col].round(3).astype(str)
        + f"<br>{y_col}: " + plot_df[y_col].round(3).astype(str)
    )

    def make_trace(label, col, visible=False):
        model_df = plot_df.dropna(subset=[col])

        return go.Scatter(
            x=model_df[x_col],
            y=model_df[y_col],
            mode="markers",
            name=label,
            visible=visible,
            marker=dict(
                size=17,
                color=model_df[col],
                colorscale=colorscale,
                cmin=-color_abs_max,
                cmax=color_abs_max,
                cmid=0,
                opacity=0.78,
                line=dict(width=0),
                colorbar=dict(
                    title=dict(text="ΔΔG", font=dict(size=20)),
                    tickfont=dict(size=16),
                ),
            ),
            text=(
                hover_base.loc[model_df.index]
                + f"<br>{label}: "
                + model_df[col].round(3).astype(str)
            ),
            hovertemplate="%{text}<extra></extra>",
        )

    traces = [
        make_trace(label, col, visible=(i == 0))
        for i, (label, col) in enumerate(models.items())
    ]

    buttons = [
        dict(
            label=label,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(models))]},
            ],
        )
        for i, label in enumerate(models)
    ]

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text="P53 Tumor Suppressor ΔΔG Model",
            font=dict(size=26),
            x=0.5,
            y=0.96,
            xanchor="center",
            yanchor="top",
        ),
        width=1000,
        height=700,
        template="plotly_white",
        font=dict(size=16),
        xaxis=dict(
            title=dict(text=x_col, font=dict(size=22)),
            tickfont=dict(size=16),
            zeroline=True,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=y_col, font=dict(size=22)),
            tickfont=dict(size=16),
            zeroline=True,
            showgrid=True,
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.93,
                y=0.91,
                xanchor="right",
                yanchor="top",
                font=dict(size=16),
                buttons=buttons,
            )
        ],
        margin=dict(l=80, r=70, t=120, b=80),
    )

    fig
    return


if __name__ == "__main__":
    app.run()
