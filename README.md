# Personal-Project
Personal Project

 
  # Market Data Analysis with Pandas

A complete data-analysis workflow in a Jupyter Notebook: load, inspect, clean, engineer features, calculate KPIs, analyse by month and sector, and visualise.

**Data:** synthetic (fictional tickers, simulated prices) with data-quality problems injected on purpose (missing values, duplicates, inconsistent text, wrong prices). The same method applies to real market data.

## Skills shown
- Pandas: `read_csv`, `info`, `describe`, `isna`, `drop_duplicates`, `to_datetime`, `groupby`, `agg`, `transform`, `rolling`, `pct_change`, `merge`, `pivot_table`, `pivot`
- Data cleaning: types, duplicates, outliers, missing values
- KPIs: total return, annualised volatility, maximum drawdown, return per risk, turnover
- Visualisation with matplotlib

## How to run
```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook market_data_analysis_pandas.ipynb
```
Then run the cells in order (Kernel > Restart & Run All). The notebook creates the `data/` folder and the CSV files by itself.

## Next steps
- Replace the synthetic data with a real public dataset and rerun the pipeline
- Turn the cleaning steps into functions with data-quality checks
- Build a Power BI dashboard on top of the clean table
