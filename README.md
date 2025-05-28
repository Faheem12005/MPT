# 📊 Modern Portfolio Theory (MPT)

This is an interactive web app built with **Streamlit** that simulates **Modern Portfolio Theory**. The app uses **Monte Carlo simulation** to generate thousands of portfolios and highlights the optimal one based on the **Sharpe Ratio**.

---

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mpt-portfolio-optimizer.git
cd mpt-portfolio-optimizer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 🛠 How It Works

1. Select:

   * Timeframe to analyze expected returns
   * Investing period for cumulative return comparison
   * Start date (restricted to valid range)
   * Tickers to include in the portfolio
   * Number of random portfolios to simulate

2. Click **Compute** to run the simulation:

   * Calculates expected returns using CAPM
   * Simulates thousands of portfolios
   * Finds the one with the highest Sharpe Ratio (Tangent Portfolio)


