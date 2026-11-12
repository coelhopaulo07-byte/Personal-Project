# Personal-Project
Personal Project

 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Market Data Analysis with Pandas\n",
    "\n",
    "**Goal:** practise a complete data-analysis workflow with Pandas, from raw and messy data to clear KPIs and insights.\n",
    "\n",
    "**What this notebook covers**\n",
    "1. Loading and inspecting raw data (`read_csv`, `info`, `describe`, `isna`)\n",
    "2. Cleaning (types, duplicates, missing values, outliers)\n",
    "3. Feature engineering (returns, moving averages, rolling volatility)\n",
    "4. KPIs per instrument with `groupby` and `agg`\n",
    "5. Monthly and sector analysis (`merge`, `pivot_table`)\n",
    "6. Visualisation (matplotlib)\n",
    "\n",
    "> **Note:** the data is **synthetic**. The tickers are fictional and the prices are simulated, with data-quality problems injected on purpose so the cleaning steps have something real to fix. The method is what matters, and it applies to real market data."
   ],
   "id": "c000"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "pd.set_option(\"display.max_columns\", 20)\n",
    "pd.set_option(\"display.float_format\", \"{:,.4f}\".format)\n",
    "os.makedirs(\"data\", exist_ok=True)"
   ],
   "id": "c001",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Create the raw dataset\n",
    "\n",
    "In a real job the data would already exist (a database, an export, a vendor file). Here we simulate **daily prices and volumes for 8 fictional tickers over 2 years**, then save them as CSV files and reload them, just like a normal workflow.\n",
    "\n",
    "We also inject typical problems: missing values, duplicate rows, inconsistent text, and a few wrong prices (typing errors)."
   ],
   "id": "c002"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "rng = np.random.default_rng(42)          # fixed seed -> same results every time\n",
    "\n",
    "tickers = {\n",
    "    \"ALFA\": \"Banks\",      \"BETA\": \"Banks\",\n",
    "    \"GAMA\": \"Energy\",     \"DELT\": \"Energy\",\n",
    "    \"EPSI\": \"Technology\", \"ZETA\": \"Technology\",\n",
    "    \"ETAX\": \"Healthcare\", \"THET\": \"Healthcare\",\n",
    "}\n",
    "dates = pd.bdate_range(\"2024-01-01\", \"2025-12-31\")   # business days only\n",
    "\n",
    "rows = []\n",
    "for ticker in tickers:\n",
    "    start_price = rng.uniform(20, 120)\n",
    "    daily_ret = rng.normal(loc=0.0003, scale=rng.uniform(0.01, 0.025), size=len(dates))\n",
    "    price = start_price * np.cumprod(1 + daily_ret)\n",
    "    volume = rng.lognormal(mean=12, sigma=0.4, size=len(dates)).astype(int)\n",
    "    rows.append(pd.DataFrame({\"date\": dates, \"ticker\": ticker, \"close\": price, \"volume\": volume}))\n",
    "\n",
    "raw = pd.concat(rows, ignore_index=True)\n",
    "\n",
    "# ---- inject data-quality problems ----\n",
    "raw.loc[rng.choice(raw.index, 40, replace=False), \"close\"] = np.nan        # missing prices\n",
    "raw.loc[rng.choice(raw.index, 25, replace=False), \"volume\"] = np.nan       # missing volumes\n",
    "bad = rng.choice(raw.index, 6, replace=False)\n",
    "raw.loc[bad, \"close\"] = raw.loc[bad, \"close\"] * 10                          # typing errors\n",
    "dups = raw.sample(30, random_state=1)\n",
    "raw = pd.concat([raw, dups], ignore_index=True)                             # duplicate rows\n",
    "raw[\"ticker\"] = raw[\"ticker\"].where(rng.random(len(raw)) > 0.05, raw[\"ticker\"].str.lower() + \" \")  # messy text\n",
    "raw[\"date\"] = raw[\"date\"].dt.strftime(\"%Y-%m-%d\")                           # dates stored as text\n",
    "\n",
    "raw = raw.sample(frac=1, random_state=7).reset_index(drop=True)             # shuffle\n",
    "raw.to_csv(\"data/market_data_raw.csv\", index=False)\n",
    "pd.DataFrame({\"ticker\": list(tickers), \"sector\": list(tickers.values())}).to_csv(\"data/sectors.csv\", index=False)\n",
    "print(\"Files saved in ./data\")"
   ],
   "id": "c003",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load and inspect"
   ],
   "id": "c004"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df = pd.read_csv(\"data/market_data_raw.csv\")\n",
    "sectors = pd.read_csv(\"data/sectors.csv\")\n",
    "df.head(10)"
   ],
   "id": "c005",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df.info()"
   ],
   "id": "c006",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "print(\"Missing values per column:\")\n",
    "print(df.isna().sum())\n",
    "print()\n",
    "print(\"Duplicate rows:\", df.duplicated().sum())\n",
    "print(\"Distinct ticker spellings:\", sorted(df[\"ticker\"].unique()))"
   ],
   "id": "c007",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "df.describe()"
   ],
   "id": "c008",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**What the inspection tells us**\n",
    "- `date` is text, not a real date.\n",
    "- `close` and `volume` have missing values.\n",
    "- There are duplicate rows.\n",
    "- Ticker names are inconsistent (lower case and trailing spaces).\n",
    "- The maximum `close` is far above the typical values, which points to wrong prices."
   ],
   "id": "c009"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Clean the data"
   ],
   "id": "c010"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "clean = df.copy()\n",
    "\n",
    "# 3.1 consistent text and real dates\n",
    "clean[\"ticker\"] = clean[\"ticker\"].str.strip().str.upper()\n",
    "clean[\"date\"] = pd.to_datetime(clean[\"date\"])\n",
    "\n",
    "# 3.2 remove duplicates\n",
    "before = len(clean)\n",
    "clean = clean.drop_duplicates(subset=[\"date\", \"ticker\"])\n",
    "print(f\"Duplicates removed: {before - len(clean)}\")  # more than before, because \"alfa \" and \"ALFA\" now count as the same ticker\n",
    "\n",
    "# 3.3 sort so that time-based calculations are correct\n",
    "clean = clean.sort_values([\"ticker\", \"date\"]).reset_index(drop=True)"
   ],
   "id": "c011",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 3.4 outliers: compare each price with the rolling median of its neighbours.\n",
    "# A price more than 3x (or less than 1/3 of) its local median is treated as a typing error.\n",
    "local_median = clean.groupby(\"ticker\")[\"close\"].transform(\n",
    "    lambda s: s.rolling(5, center=True, min_periods=1).median()\n",
    ")\n",
    "ratio = clean[\"close\"] / local_median\n",
    "outliers = (ratio > 3) | (ratio < 1 / 3)\n",
    "print(\"Outliers found:\", int(outliers.sum()))\n",
    "clean.loc[outliers, \"close\"] = np.nan"
   ],
   "id": "c012",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 3.5 missing values\n",
    "# prices: interpolate inside each ticker (never mix tickers)\n",
    "clean[\"close\"] = clean.groupby(\"ticker\")[\"close\"].transform(\n",
    "    lambda s: s.interpolate().ffill().bfill()\n",
    ")\n",
    "# volumes: use the median volume of the same ticker\n",
    "clean[\"volume\"] = clean.groupby(\"ticker\")[\"volume\"].transform(lambda s: s.fillna(s.median()))\n",
    "\n",
    "print(\"Missing values after cleaning:\")\n",
    "print(clean.isna().sum())\n",
    "print(\"Rows:\", len(clean), \"| Tickers:\", clean[\"ticker\"].nunique())"
   ],
   "id": "c013",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature engineering"
   ],
   "id": "c014"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "g = clean.groupby(\"ticker\")\n",
    "\n",
    "clean[\"daily_return\"] = g[\"close\"].pct_change()\n",
    "clean[\"ma_20\"] = g[\"close\"].transform(lambda s: s.rolling(20).mean())\n",
    "clean[\"vol_20\"] = clean.groupby(\"ticker\")[\"daily_return\"].transform(lambda s: s.rolling(20).std())\n",
    "clean[\"turnover\"] = clean[\"close\"] * clean[\"volume\"]\n",
    "\n",
    "# add the sector with a merge (like a SQL LEFT JOIN)\n",
    "clean = clean.merge(sectors, on=\"ticker\", how=\"left\")\n",
    "clean.head()"
   ],
   "id": "c015",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. KPIs per ticker"
   ],
   "id": "c016"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "def max_drawdown(prices):\n",
    "    \"\"\"Largest fall from a previous peak (a negative number).\"\"\"\n",
    "    return (prices / prices.cummax() - 1).min()\n",
    "\n",
    "kpis = clean.groupby([\"ticker\", \"sector\"]).agg(\n",
    "    first_price=(\"close\", \"first\"),\n",
    "    last_price=(\"close\", \"last\"),\n",
    "    avg_daily_volume=(\"volume\", \"mean\"),\n",
    "    avg_daily_turnover=(\"turnover\", \"mean\"),\n",
    "    daily_vol=(\"daily_return\", \"std\"),\n",
    ").reset_index()\n",
    "\n",
    "kpis[\"total_return\"] = kpis[\"last_price\"] / kpis[\"first_price\"] - 1\n",
    "kpis[\"annual_volatility\"] = kpis[\"daily_vol\"] * np.sqrt(252)\n",
    "kpis[\"max_drawdown\"] = kpis[\"ticker\"].map(clean.groupby(\"ticker\")[\"close\"].apply(max_drawdown))\n",
    "kpis[\"return_per_risk\"] = kpis[\"total_return\"] / kpis[\"annual_volatility\"]\n",
    "\n",
    "kpis = kpis.drop(columns=[\"daily_vol\", \"first_price\", \"last_price\"]).sort_values(\"total_return\", ascending=False)\n",
    "kpis"
   ],
   "id": "c017",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Monthly and sector analysis"
   ],
   "id": "c018"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "clean[\"month\"] = clean[\"date\"].dt.to_period(\"M\")\n",
    "\n",
    "monthly_sector = (\n",
    "    clean.groupby([\"month\", \"sector\"])\n",
    "         .agg(avg_turnover=(\"turnover\", \"mean\"), avg_return=(\"daily_return\", \"mean\"))\n",
    "         .reset_index()\n",
    ")\n",
    "\n",
    "turnover_pivot = monthly_sector.pivot_table(index=\"month\", columns=\"sector\", values=\"avg_turnover\")\n",
    "turnover_pivot.tail()"
   ],
   "id": "c019",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "sector_summary = clean.groupby(\"sector\").agg(\n",
    "    avg_turnover=(\"turnover\", \"mean\"),\n",
    "    avg_daily_return=(\"daily_return\", \"mean\"),\n",
    "    volatility=(\"daily_return\", \"std\"),\n",
    ").sort_values(\"avg_turnover\", ascending=False)\n",
    "sector_summary"
   ],
   "id": "c020",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Visualisation"
   ],
   "id": "c021"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "\n",
    "# (a) price and 20-day moving average of the best performer\n",
    "best = kpis.iloc[0][\"ticker\"]\n",
    "d = clean[clean[\"ticker\"] == best]\n",
    "axes[0, 0].plot(d[\"date\"], d[\"close\"], label=\"Close\")\n",
    "axes[0, 0].plot(d[\"date\"], d[\"ma_20\"], label=\"20-day MA\")\n",
    "axes[0, 0].set_title(f\"{best}: price and 20-day moving average\")\n",
    "axes[0, 0].legend()\n",
    "\n",
    "# (b) total return by ticker\n",
    "axes[0, 1].bar(kpis[\"ticker\"], kpis[\"total_return\"] * 100)\n",
    "axes[0, 1].set_title(\"Total return by ticker (%)\")\n",
    "\n",
    "# (c) monthly average turnover by sector\n",
    "turnover_pivot.index = turnover_pivot.index.to_timestamp()\n",
    "turnover_pivot.plot(ax=axes[1, 0])\n",
    "axes[1, 0].set_title(\"Average daily turnover by sector (monthly)\")\n",
    "axes[1, 0].set_xlabel(\"\")\n",
    "\n",
    "# (d) rolling volatility for two tickers\n",
    "for t in kpis[\"ticker\"].iloc[[0, -1]]:\n",
    "    x = clean[clean[\"ticker\"] == t]\n",
    "    axes[1, 1].plot(x[\"date\"], x[\"vol_20\"], label=t)\n",
    "axes[1, 1].set_title(\"20-day rolling volatility (best vs worst performer)\")\n",
    "axes[1, 1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ],
   "id": "c022",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# correlation of daily returns between tickers\n",
    "returns_wide = clean.pivot(index=\"date\", columns=\"ticker\", values=\"daily_return\")\n",
    "corr = returns_wide.corr()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "im = ax.imshow(corr, vmin=-1, vmax=1, cmap=\"coolwarm\")\n",
    "ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns)\n",
    "ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)\n",
    "fig.colorbar(im)\n",
    "ax.set_title(\"Correlation of daily returns\")\n",
    "plt.show()"
   ],
   "id": "c023",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Key findings"
   ],
   "id": "c024"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "best = kpis.iloc[0]\n",
    "worst = kpis.iloc[-1]\n",
    "most_volatile = kpis.sort_values(\"annual_volatility\", ascending=False).iloc[0]\n",
    "top_sector = sector_summary.index[0]\n",
    "deepest_dd = kpis.sort_values(\"max_drawdown\").iloc[0]\n",
    "\n",
    "print(f\"Best performer:      {best['ticker']} ({best['total_return']:.1%})\")\n",
    "print(f\"Worst performer:     {worst['ticker']} ({worst['total_return']:.1%})\")\n",
    "print(f\"Most volatile:       {most_volatile['ticker']} (annualised volatility {most_volatile['annual_volatility']:.1%})\")\n",
    "print(f\"Deepest drawdown:    {deepest_dd['ticker']} ({deepest_dd['max_drawdown']:.1%})\")\n",
    "print(f\"Highest turnover:    {top_sector} sector\")"
   ],
   "id": "c025",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Conclusions and next steps\n",
    "\n",
    "**What I did:** took a messy dataset, cleaned it step by step (types, duplicates, outliers, missing values), built features, calculated KPIs per ticker and per sector, and visualised the results.\n",
    "\n",
    "**Ideas to take it further**\n",
    "- Replace the synthetic data with a **real public dataset** (for example historical prices exported from a finance website or a Kaggle dataset) and rerun the same pipeline.\n",
    "- Wrap the cleaning steps in **functions** and add checks that stop the process if data quality is bad.\n",
    "- Add a **Power BI dashboard** on top of the clean table (`clean.to_csv(...)`) with the same KPIs.\n",
    "- Automate the monthly report with a script."
   ],
   "id": "c026"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
