📊 Stock Price Forecasting using ARIMA – ADANIGREEN
📌 Overview
This project performs time series analysis and forecasting on the stock ADANIGREEN using the ARIMA (AutoRegressive Integrated Moving Average) model.

The objective is to analyze historical stock prices and predict future trends based on statistical modeling.

📂 Dataset
Source: NSE India (National Stock Exchange)

Stock: ADANIPOWER

Duration: Past 1 year

Features used:

Date
Closing Price
⚙️ Methodology
1. Data Preprocessing
Converted date column to datetime format
Sorted data chronologically
Set date as index
Handled missing values using interpolation
Ensured regular business-day frequency
2. Exploratory Data Analysis
Visualized closing price trend
Generated differenced series
Plotted ACF and PACF graphs
3. Stationarity Check
Applied Augmented Dickey-Fuller (ADF) Test
Result:

p-value > 0.05 → Data is non-stationary
Applied first-order differencing (d = 1)
4. ARIMA Model
Performed grid search to find optimal (p, d, q)
Selected model based on lowest AIC value
Trained ARIMA model on historical data
5. Model Evaluation
Used train-test split

Metrics:

MAE (Mean Absolute Error)
RMSE (Root Mean Square Error)
MAPE (Mean Absolute Percentage Error)
6. Forecasting
Forecasted next 30 business days
Visualized forecast alongside historical data


📊 Observations & Interpretation
The stock price shows high volatility, typical of energy sector stocks.
ADF test confirmed the data was non-stationary, requiring differencing.
ACF and PACF plots helped determine ARIMA parameters.
The ARIMA model captured the general trend but not sharp fluctuations.
🔮 Forecast Insight:
The model suggests a moderate trend (slightly upward or stable depending on output).
No extreme spikes or crashes are predicted in the next 30 days.
Predictions are smooth due to ARIMA limitations.
⚠️ Limitations
ARIMA assumes linear patterns and cannot capture sudden market shocks.
External factors like news, policies, or global events are not considered.
Forecast accuracy decreases over longer horizons.
📑 Summary Report
This project focuses on forecasting the stock price of ADANIPOWER using the ARIMA model. Historical daily closing price data for the past year was collected from NSE India.

The dataset was preprocessed by converting the date column into datetime format, handling missing values, and ensuring a continuous time series. Visualization of the closing price showed noticeable fluctuations, indicating volatility in the stock.

The Augmented Dickey-Fuller (ADF) test was conducted to check for stationarity. Since the p-value was greater than 0.05, the data was non-stationary and required differencing. First-order differencing was applied.

ACF and PACF plots were analyzed to determine suitable ARIMA parameters. A grid search approach was used to identify the best model based on AIC.

The ARIMA model was trained and evaluated using standard error metrics such as MAE, RMSE, and MAPE. The model performed reasonably well in capturing the overall trend but struggled with sudden fluctuations.

Finally, a 30-day forecast was generated. The forecast indicates a relatively stable to very slightly upward trend, with no major volatility expected in the short term.

🤖 AI Ethics & Responsible Usage Declaration
This project uses statistical and AI-based forecasting techniques strictly for educational purposes. The predictions generated are not financial advice and should not be used for investment decisions.

All data used is publicly available from NSE India.

📚 References
NSE India Official Website
Statsmodels Documentation
Time Series Forecasting (Box-Jenkins Methodology)
