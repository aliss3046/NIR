````markdown
# Grain Market Data Analysis and Price Forecasting

This project focuses on the analysis and forecasting of grain crop prices using open international statistical data.

## Project Overview

The main goal of the project is to study the dynamics of grain crop prices and build forecasting models based on historical data. The project combines data on grain production and commodity prices, performs exploratory data analysis, and compares several forecasting approaches.

## Data Sources

The dataset was created using open data from:

- FAOSTAT — grain crop production data
- World Bank Commodity Markets — commodity and grain price data

The final dataset covers the period from 2000 to 2024 and includes information about:

- wheat
- maize
- barley
- sorghum
- rice
- oil prices
- production indicators

## Methods Used

The project includes:

- data cleaning and preprocessing
- aggregation of monthly price data into yearly indicators
- descriptive statistics
- correlation analysis
- time series analysis
- price forecasting
- feature importance analysis

The following models were implemented and compared:

- Linear Regression
- ARIMA
- Random Forest

## Main Results

The results showed that there is no single universal model that performs best for all crops.

Best models by crop:

| Crop | Best Model | MAE |
|---|---|---|
| Wheat | Linear Regression | 16.18 |
| Maize | Random Forest | 6.72 |
| Barley | Linear Regression | 20.09 |
| Sorghum | Random Forest | 11.01 |
| Rice | Linear Regression | 8.70 |

ARIMA showed the weakest results because it uses only historical price values and does not take into account external factors such as oil prices, production volumes, and prices of related crops.

## Forecast

Using the selected models, a scenario forecast of grain crop prices was built for the period from 2025 to 2030. The forecast suggests moderate price growth for most analyzed crops, assuming that historical trends continue and no major external shocks occur.

## Technologies

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Project Structure

```text
.
├── data/              # Source and processed datasets
├── notebooks/         # Jupyter notebooks
├── src/               # Python scripts
├── results/           # Charts and model outputs
└── README.md
````

## Conclusion

The project demonstrates how statistical methods and machine learning models can be used to analyze grain markets, identify relationships between economic indicators, and build price forecasts for agricultural commodities.

```
```
