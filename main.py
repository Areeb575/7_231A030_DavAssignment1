from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.tseries.offsets import BDay
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# Settings
# ----------------------------
CSV_PATH = "adanigreen.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("default")


# ----------------------------
# Helpers
# ----------------------------
def save_fig(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()

    if "DATE" not in df.columns or "CLOSE" not in df.columns:
        raise ValueError(f"Expected DATE and CLOSE columns. Found: {df.columns.tolist()}")

    # Parse date and close price
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")

    # Clean data
    df = df.dropna(subset=["DATE", "CLOSE"])
    df = df.drop_duplicates(subset=["DATE"])
    df = df.sort_values("DATE")
    df = df.set_index("DATE")

    # Force a regular business-day frequency so statsmodels gets a valid time index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    df = df.reindex(full_index)
    df.index.name = "DATE"

    # Fill missing business days safely
    df["CLOSE"] = df["CLOSE"].interpolate(method="time").ffill().bfill()

    return df


def adf_test(series):
    result = adfuller(series, autolag="AIC")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Used Lag:", result[2])
    print("Observations:", result[3])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value}")
    return result[1]


def save_trend_plot(series):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values)
    ax.set_title("ADANIPOWER Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    fig.autofmt_xdate()
    save_fig(fig, "closing_price_trend.png")


def save_acf_pacf_plots(series):
    lags = min(40, len(series) // 2 - 1)
    lags = max(lags, 10)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(series, ax=ax, lags=lags)
    ax.set_title("ACF Plot")
    save_fig(fig, "acf_plot.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_pacf(series, ax=ax, lags=lags, method="ywm")
    ax.set_title("PACF Plot")
    save_fig(fig, "pacf_plot.png")


def grid_search_arima(train_series, d, p_values=range(0, 4), q_values=range(0, 4)):
    best_aic = np.inf
    best_order = None
    best_fit = None

    for p in p_values:
        for q in q_values:
            try:
                model = ARIMA(
                    train_series,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
                    best_fit = fit
            except Exception:
                continue

    if best_fit is None:
        raise RuntimeError("No ARIMA model could be fit. Try different p/d/q ranges.")

    return best_order, best_fit


def evaluate_forecast(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print("\nEvaluation on test set:")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return mae, rmse, mape


def save_forecast_plot(history, forecast):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.index, history.values, label="Historical")
    ax.plot(forecast.index, forecast.values, label="Forecast", color="red")
    ax.set_title("30-Day Forecast for ADANIPOWER")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()
    save_fig(fig, "forecast_plot.png")


def save_differenced_plot(series):
    diff = series.diff().dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(diff.index, diff.values)
    ax.set_title("Differenced Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Difference in Close Price")
    fig.autofmt_xdate()
    save_fig(fig, "differenced_series.png")


# ----------------------------
# Main workflow
# ----------------------------
df = load_and_prepare_data(CSV_PATH)
close_series = df["CLOSE"]

print("Data shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())

# Save preprocessing graphs
save_trend_plot(close_series)
save_differenced_plot(close_series)
save_acf_pacf_plots(close_series)

# Stationarity check
p_value = adf_test(close_series)

# Choose differencing based on ADF
d = 0 if p_value <= 0.05 else 1
print(f"\nChosen differencing order d = {d}")

# Train-test split for evaluation
test_size = min(30, max(10, len(close_series) // 5))
train = close_series.iloc[:-test_size]
test = close_series.iloc[-test_size:]

# Fit best ARIMA model on train set
best_order, best_fit = grid_search_arima(train, d=d, p_values=range(0, 4), q_values=range(0, 4))
print(f"\nBest ARIMA order from AIC search: {best_order}")
print(best_fit.summary())

# Evaluate on test set
test_forecast = best_fit.forecast(steps=len(test))
test_forecast.index = test.index
evaluate_forecast(test, test_forecast)

# Refit on full data using best order
final_model = ARIMA(
    close_series,
    order=best_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_fit = final_model.fit()

# Forecast next 30 business days
future_steps = 30
future_forecast = final_fit.forecast(steps=future_steps)
future_index = pd.bdate_range(start=close_series.index[-1] + BDay(1), periods=future_steps)
future_forecast = pd.Series(future_forecast.values, index=future_index, name="Forecast")

# Save final forecast plot
history_for_plot = close_series.iloc[-120:] if len(close_series) > 120 else close_series
save_forecast_plot(history_for_plot, future_forecast)

print("\nFuture forecast:")
print(future_forecast)