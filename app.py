import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta


st.title("Modern Portfolio Theory")

class MPT:
    def __init__(self, relative_ticker, asset_tickers, start_date, period, invest_period, portfolio_count):
        self.start_date = start_date
        self.end_date = start_date + relativedelta(years=period)
        self.invest_period = invest_period
        self.asset_tickers = asset_tickers
        self.results = []
        self.count = portfolio_count
        self.asset_data = yf.Tickers(" ".join(self.asset_tickers)).history(end=f"{self.end_date}", interval='1d', auto_adjust=False, start=f"{start_date}")

        risk_free_returns = yf.Ticker('^IRX').history(end=f"{self.end_date}", interval='1d', auto_adjust=False, start=f"{start_date}")
        self.risk_free_rate = risk_free_returns['Close'].dropna().mean() / 100

        data = yf.Ticker(f"{relative_ticker}").history(end=f"{self.end_date}", interval='1d', auto_adjust=False, start=f"{start_date}")
        data.index = data.index.tz_localize(None)
        self.market_returns = data['Adj Close'].pct_change().dropna()

    def expected_return(self, ticker):
        returns = self.asset_data['Adj Close'][ticker].pct_change().dropna()
        beta = returns.cov(self.market_returns) / self.market_returns.var()
        expected_return = self.risk_free_rate + beta * (self.market_returns.mean() * 252 - self.risk_free_rate)
        return expected_return

    def portfolio_volatility(self, weights):
        stddev = 0
        returns_data = self.asset_data['Adj Close'].pct_change().dropna()       
        for i in range(len(weights)):
            for j in range(len(weights)):
                stddev += weights[i] * weights[j] * returns_data[self.asset_tickers[i]].cov(returns_data[self.asset_tickers[j]])
        return np.sqrt(stddev * 252)
    
    def mpt(self):
        portfolios = self.count
        individual_returns = [self.expected_return(ticker) for ticker in self.asset_tickers]

        latest_iteration = st.empty()
        bar = st.progress(0.00)
        for i in range(portfolios):
            #randomly get portfolio weights
            weights = np.random.dirichlet(np.ones(len(self.asset_tickers)))
            risk_stddev = self.portfolio_volatility(weights)
            exp_return = np.dot(weights, individual_returns)
            sharpe_ratio = (exp_return - self.risk_free_rate) / risk_stddev
            self.results.append([weights, risk_stddev, exp_return, sharpe_ratio])
            latest_iteration.text(f"Portfolio {i + 1}")
            bar.progress((i + 1) / portfolios)

        latest_iteration.text("Done!")
        df = pd.DataFrame(self.results, columns=['weights', 'Risk (StdDev)', 'Expected Return', 'Sharpe Ratio'])
        weights_df = pd.DataFrame(df['weights'].tolist(), columns=[f'Weight_{ticker}' for ticker in self.asset_tickers])
        df_final = pd.concat([weights_df, df.drop(columns=['weights'])], axis=1)
        best_portfolio = df_final.loc[df_final['Sharpe Ratio'].idxmax()]
        self.best_portfolio = best_portfolio
    
    def plot_returns(self):
        data = yf.Tickers((" ".join(self.asset_tickers))).history(period=f"{self.invest_period}y", interval='1d', auto_adjust=False)
        daily_returns = data['Adj Close'].pct_change().dropna()
        weights = np.array([self.best_portfolio[f"Weight_{ticker}"] for ticker in self.asset_tickers])
        portfolio_daily_returns = daily_returns.dot(weights)
        portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod() - 1
        data = yf.Tickers('^GSPC').history(period=f"{self.invest_period}y", interval='1d', auto_adjust=False)
        returns_data_market = data['Adj Close'].pct_change().dropna()   
        returns_data_market = (1 + returns_data_market).cumprod() - 1
        returns_data_market['Portfolio'] = portfolio_cum_returns
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=returns_data_market.index, y=returns_data_market['Portfolio'], name='Tangent Portfolio'))
        fig.add_trace(go.Scatter(x=returns_data_market.index, y=returns_data_market['^GSPC'], name='S&P 500'))
        fig.update_layout(title="Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")    
        return fig
    
    def efficient_frontier(self):
        risks = [r[1] for r in self.results]
        returns = [r[2] for r in self.results]
        sharpes = [r[3] for r in self.results]
        weights = [r[0] for r in self.results]

        custom_data = [
            [f"{w:.2%}" for w in weight] + [f"{ret:.2%}", f"{risk:.2%}", f"{sharpe:.2f}"]
            for weight, ret, risk, sharpe in zip(weights, returns, risks, sharpes)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers',
            marker=dict(color=sharpes, colorscale='Viridis', size=7, showscale=True, colorbar=dict(title="Sharpe")),
            customdata=custom_data,
            hovertemplate=(
                "<b>Weights:</b><br>" +
                "<br>".join([f"{ticker}: %{{customdata[{i}]}}" for i, ticker in enumerate(self.asset_tickers)]) +
                "<br><b>Return:</b> %{customdata[" + str(len(self.asset_tickers)) + "]}" +
                "<br><b>Risk:</b> %{customdata[" + str(len(self.asset_tickers)+1) + "]}" +
                "<br><b>Sharpe:</b> %{customdata[" + str(len(self.asset_tickers)+2) + "]}"
            ),
            name="Portfolios"
        ))

        fig.add_trace(go.Scatter(
            x=[self.best_portfolio['Risk (StdDev)']],
            y=[self.best_portfolio['Expected Return']],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='x'),
            text=["Tangent Portfolio"],
            textposition="top center",
            name="Tangent Portfolio"
        ))

        fig.update_layout(
            title="Efficient Frontier (Modern Portfolio Theory)",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            legend=dict(x=0.01, y=0.99)
        )

        return fig

    def plot_best_portfolio_pie(self):
        weights = {ticker: self.best_portfolio[f'Weight_{ticker}'] for ticker in self.asset_tickers}
        weights = {k: v for k, v in weights.items() if v > 0.01}  
        fig = px.pie(
            names=list(weights.keys()),
            values=list(weights.values()),
            title="Tangent Portfolio Allocation",
            hole=0.3 
        )
        fig.update_layout(title="Portfolio Distribution")
        return fig
    
with st.sidebar:
    period = st.slider(min_value=1, max_value=10, label="Enter time frame to calculate expected returns (years):")
    invest_period = st.slider(min_value=1, max_value=10, label="Enter investing time frame to check cumulative returns (years):")
    max_start_date = date.today() - relativedelta(years=period)
    start_period = st.date_input("Enter start period", max_value=max_start_date)
    asset_tickers = st.multiselect(
        label="Select Asset Tickers",
        options=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'UNH', 'HD', 'MA', 'PG', 'XOM', 'BAC', 'PFE', 'LLY', 'NFLX'
        ],
        default=['AAPL', 'MSFT', 'GOOGL']
    )

count = st.number_input(label="Enter no of Portfolios", min_value=500, max_value=10000, step=50)
mpt = MPT('^GSPC', asset_tickers, start_date=start_period, period=period, invest_period=invest_period, portfolio_count=count)
if st.button('Compute'):
    mpt.mpt()
    fig1 = mpt.plot_returns() #plotly chart 
    fig2 = mpt.efficient_frontier()
    st.plotly_chart(fig2)    
    st.plotly_chart(fig1)
    st.subheader("Tangent Portfolio Info")
    st.table(mpt.best_portfolio)
    fig3 = mpt.plot_best_portfolio_pie()
    st.plotly_chart(fig3)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Assumptions")
    st.markdown("- **Returns are normally distributed** — Markets follow a bell curve; outliers are rare.")
    st.markdown("- **Investors are rational** — Decisions are logical and risk-averse, not emotional.")
    st.markdown("- **Equal access to information** — No unfair advantage; everyone sees the same data.")
    st.markdown("- **Individual investors don't move markets** — Prices reflect the collective actions of all participants.")

with col2:
    st.subheader("Pitfalls")
    st.markdown("- **Past ≠ Future** — Historical returns don’t guarantee future performance.")
    st.markdown("- **Markets aren't always normal** — Real returns often defy the bell curve, especially in crises.")
    st.markdown("- **Static assumptions** — MPT doesn’t adapt well to evolving markets or investor needs.")
    st.markdown("- **Not all investors are rational** — Emotions, bias, and noise affect real-world decisions.")
