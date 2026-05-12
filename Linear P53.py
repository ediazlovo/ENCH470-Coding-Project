# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.5",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.7.0",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    return (
        LinearRegression,
        Pipeline,
        StandardScaler,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        px,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Read CSV
    """)
    return


@app.cell
def _(pd):
    ProteinProperties = pd.read_csv("P53.csv")
    ProteinProperties
    return (ProteinProperties,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Select Features and Target
    Extracts the features from the CSV file

    Missing entries are ignored
    """)
    return


@app.cell
def _(ProteinProperties, mo, pd):
    df = ProteinProperties.copy()
    df.columns = df.columns.str.strip()

    feature_cols = ["Residue Depth", "CA Depth", "RSA"]
    target_col = "DDG"
    label_col = "ID"

    model_data = df[feature_cols + [target_col, label_col]].copy()

    for col in feature_cols + [target_col]:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    model_data = model_data.dropna()

    X = model_data[feature_cols]
    y = model_data[target_col]
    mutation_labels = model_data[label_col]

    mo.vstack([
        mo.md(f"Using **{len(model_data)} rows** for Linear regression."),
        mo.ui.table(model_data.head(10)),
    ])
    return X, feature_cols, mutation_labels, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Train Linear Regression

    80% of the data is used for training

    20% of the data is used for testing

    random_rate is used for reproducing the same model results

    Linear regression is used with sklearn
    """)
    return


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    X,
    mutation_labels,
    train_test_split,
    y,
):
    X_train, X_test, y_train, y_test, mutations_train, mutations_test = train_test_split(
        X,
        y,
        mutation_labels,
        test_size=0.2,
        random_state=42,
    )

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression()),
    ])

    linear_model.fit(X_train, y_train)

    y_train_pred = linear_model.predict(X_train)
    y_test_pred = linear_model.predict(X_test)

    linear = linear_model.named_steps["linear"]
    scaler = linear_model.named_steps["scaler"]
    return (
        linear,
        linear_model,
        mutations_test,
        mutations_train,
        scaler,
        y_test,
        y_test_pred,
        y_train,
        y_train_pred,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Model Performance
    Model is assessed by finding the R^2, Root Mean Squared Error, and Mean Absolute Error
    """)
    return


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    mo,
    pd,
    r2_score,
    y_test,
    y_test_pred,
    y_train,
    y_train_pred,
):
    metrics = pd.DataFrame({
        "Split": ["Train", "Test"],
        "R2": [
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred),
        ],
        "MAE": [
            mean_absolute_error(y_train, y_train_pred),
            mean_absolute_error(y_test, y_test_pred),
        ],
        "RMSE": [
            mean_squared_error(y_train, y_train_pred) ** 0.5,
            mean_squared_error(y_test, y_test_pred) ** 0.5,
        ],
    })

    mo.ui.table(metrics.round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Linear Coefficents
    Determiens Linear coefficients from:
    $$
    \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
    $$

    Outputs after applying coefficient penalty (Ordinary Least Squares)
    $$
    \text{Loss} =
    \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
    $$
    """)
    return


@app.cell
def _(feature_cols, linear, mo, np, pd, scaler):
    coef_standardized = pd.DataFrame({
        "Feature": feature_cols,
        "Standardized coefficient": linear.coef_,
    })

    coef_original_scale = linear.coef_ / scaler.scale_
    intercept_original_scale = linear.intercept_ - np.sum(
        linear.coef_ * scaler.mean_ / scaler.scale_
    )

    coef_original = pd.DataFrame({
        "Feature": feature_cols,
        "Original-scale coefficient": coef_original_scale,
    })

    mo.vstack([
        mo.md("Standardized coefficients:"),
        mo.ui.table(coef_standardized.round(4)),
        mo.md(f"Original-scale intercept: **{intercept_original_scale:.4f}**"),
        mo.ui.table(coef_original.round(4)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Interactive Predicted vs Experimental ΔΔG Plot
    Interactive Plot

    Hovering over data points reveals data parameters and source mutation ID
    """)
    return


@app.cell
def _(
    mutations_test,
    mutations_train,
    pd,
    px,
    y_test,
    y_test_pred,
    y_train,
    y_train_pred,
):
    plot_df = pd.concat([
        pd.DataFrame({
            "Experimental ΔΔG": y_train,
            "Predicted ΔΔG": y_train_pred,
            "ID": mutations_train,
            "Split": "Train",
        }),
        pd.DataFrame({
            "Experimental ΔΔG": y_test,
            "Predicted ΔΔG": y_test_pred,
            "ID": mutations_test,
            "Split": "Test",
        }),
    ])

    min_val = min(plot_df["Experimental ΔΔG"].min(), plot_df["Predicted ΔΔG"].min())
    max_val = max(plot_df["Experimental ΔΔG"].max(), plot_df["Predicted ΔΔG"].max())

    fig = px.scatter(
        plot_df,
        x="Experimental ΔΔG",
        y="Predicted ΔΔG",
        color="Split",
        hover_name="ID",
        hover_data={
            "Experimental ΔΔG": ":.3f",
            "Predicted ΔΔG": ":.3f",
            "Split": True,
        },
        title="Linear Regression: Experimental vs Predicted ΔΔG",
    )

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="black", dash="dash"),
    )

    fig.update_layout(
        width=700,
        height=550,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Experimental ΔΔG",
        yaxis_title="Predicted ΔΔG",
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Predict New ΔΔG
    Interactive feature slider inputs
    """)
    return


@app.cell
def _(X, mo):
    residue_depth_input = mo.ui.number(
        value=float(X["Residue Depth"].median()),
        label="Residue Depth",
    )

    ca_depth_input = mo.ui.number(
        value=float(X["CA Depth"].median()),
        label="CA Depth",
    )

    rsa_input = mo.ui.number(
        value=float(X["RSA"].median()),
        label="RSA",
    )

    mo.vstack([residue_depth_input, ca_depth_input, rsa_input])
    return ca_depth_input, residue_depth_input, rsa_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #New Prediction Output
    Output ΔΔG from slider inputes using Linear Regression model
    """)
    return


@app.cell
def _(ca_depth_input, linear_model, mo, pd, residue_depth_input, rsa_input):
    new_sample = pd.DataFrame([{
        "Residue Depth": residue_depth_input.value,
        "CA Depth": ca_depth_input.value,
        "RSA": rsa_input.value,
    }])

    predicted_ddg = linear_model.predict(new_sample)[0]

    mo.md(f"Predicted Linear ΔΔG: **{predicted_ddg:.3f}**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Export CSV with Linear Predictions
    Uses features from original CSV file and runs the Linear model

    Creates a new column for prediction results

    Outputs a download for new CSV file
    """)
    return


@app.cell
def _(ProteinProperties, feature_cols, linear_model, mo, np, pd):
    ProteinProperties_with_Linear = ProteinProperties.copy()
    ProteinProperties_with_Linear.columns = ProteinProperties_with_Linear.columns.str.strip()

    linear_input = ProteinProperties_with_Linear[feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    ProteinProperties_with_Linear["Linear"] = np.nan

    valid_rows = linear_input.notna().all(axis=1)
    ProteinProperties_with_Linear.loc[valid_rows, "Linear"] = linear_model.predict(
        linear_input.loc[valid_rows]
    )

    ProteinProperties_with_Linear.to_csv("P53_with_LinearRegression_predictions.csv", index=False)

    mo.vstack([
        mo.md(
            f"Exported **P53_with_LinearRegression_predictions.csv** with Linear Regression predictions for "
            f"**{valid_rows.sum()} rows**."
        ),
        mo.ui.table(ProteinProperties_with_Linear.head(83)),
    ])
    return


if __name__ == "__main__":
    app.run()
