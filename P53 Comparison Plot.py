# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.7.0",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    #Imports
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt


    from sklearn.linear_model import LassoCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ENCH 470 Group C: P53 DUET DDG Comparison with Random Forest Prediction
    Code that takes the generated table of predictions of all regression models, reads the random forest, DUET, and mCSM columns, and plots them against experimental ddg values to determine the comparitive accuracy of each predicted value.
    """)
    return


@app.cell
def _(np, pd, plt):
    # Read the CSV file
    dfp = pd.read_csv("P53 Predictions.csv")

    # Column indices (Python uses 0-based indexing)
    x_col_index = 7          # 8th column
    y_col_indices = [13, 15, 19]  # 14th, 16th, and 20th columns

    # Get column names
    x_column = dfp.columns[x_col_index]
    y_columns = [dfp.columns[i] for i in y_col_indices]
    plt.figure(figsize=(6,6), dpi=100)


    # Plot each selected column against the 8th column
    for column in y_columns:
        fig2=plt.scatter(dfp[x_column], dfp[column], label=column)
    x = np.linspace(-8, 4, 100)
    plt.plot(x, x, 'k--')
    # Labels and title
    plt.xlabel("Experimental ΔΔG (kcal/mol)")
    plt.ylabel("Predicted ΔΔG (kcal/mol)")
    #plt.title("Comparison of ΔΔG's")
    plt.xlim(-8, 4)
    plt.ylim(-8, 4)
    plt.gca().set_aspect('equal', adjustable='box')


    # Legend and grid
    plt.legend()
    #plt.grid(True)


    # Show the plot
    fig2

    return


if __name__ == "__main__":
    app.run()
