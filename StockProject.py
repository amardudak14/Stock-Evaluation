import yfinance as yf
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from textblob import TextBlob

# Optional: Add your free NewsAPI key here (get one at https://newsapi.org/)
NEWS_API_KEY = ""  # Add your key here to enable sentiment analysis

# ----------------------- FCF CALCULATION -----------------------
def safe_get(statement, possible_labels, col):
    for label in possible_labels:
        if label in statement.index:
            return statement.loc[label][col]
    return 0

def calculate_fcf(ticker):
    company = yf.Ticker(ticker)
    try:
        income_stmt = company.financials
        cashflow_stmt = company.cashflow
        latest_year = income_stmt.columns[0]

        ebit = safe_get(income_stmt, ["Operating Income", "EBIT"], latest_year)
        depreciation = safe_get(cashflow_stmt, ["Depreciation", "Depreciation & Amortization"], latest_year)
        capex = safe_get(cashflow_stmt, ["Capital Expenditures", "Capital expenditure"], latest_year)
        wc_change = safe_get(cashflow_stmt, ["Change in Working Capital", "Changes in working capital"], latest_year)

        fcf = ebit + depreciation - capex - wc_change

        print(f"\n✅ FCF Components for {ticker}:")
        print(f"  EBIT: {ebit:,.0f}")
        print(f"  Depreciation: {depreciation:,.0f}")
        print(f"  CapEx: {capex:,.0f}")
        print(f"  Change in WC: {wc_change:,.0f}")
        print(f"  ➤ Free Cash Flow: {fcf:,.0f}\n")

        return fcf

    except Exception as e:
        print(f"❌ Error calculating FCF for {ticker}: {e}")
        return None

# ----------------------- DCF VALUATION -----------------------
def discounted_cash_flow_analysis(ticker, years=5, discount_rate=0.10, growth_rate=0.05):
    fcf = calculate_fcf(ticker)
    if fcf is None or fcf <= 0:
        return None

    future_fcfs = [fcf * (1 + growth_rate)**i for i in range(1, years + 1)]
    discounted_fcfs = [f / (1 + discount_rate)**i for i, f in enumerate(future_fcfs, 1)]

    terminal_value = future_fcfs[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
    discounted_terminal_value = terminal_value / (1 + discount_rate)**years

    intrinsic_value = sum(discounted_fcfs) + discounted_terminal_value

    company = yf.Ticker(ticker)
    market_cap = company.info.get('marketCap', 0)

    upside = ((intrinsic_value - market_cap) / market_cap) * 100 if market_cap > 0 else 0

    return intrinsic_value, market_cap, upside, discounted_fcfs, discounted_terminal_value

# ----------------------- NEWS + SENTIMENT -----------------------
def get_security_news(ticker):
    if not NEWS_API_KEY:
        print("🔒 No News API key provided. Skipping sentiment analysis.")
        return []

    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ News API error: {response.status_code}")
            return []

        data = response.json()
        articles = data.get("articles", [])
        return [article["title"] + ". " + article.get("description", "") for article in articles[:5]]

    except Exception as e:
        print(f"❌ Error retrieving news: {e}")
        return []

def analyze_sentiment(news_list):
    if not news_list:
        return 0
    sentiments = [TextBlob(article).sentiment.polarity for article in news_list]
    avg_sentiment = sum(sentiments) / len(sentiments)
    return avg_sentiment

# ----------------------- EVALUATION + VISUALIZATION -----------------------
def evaluate_security(ticker):
    print(f"\n📊 Evaluating: {ticker}")
    print(f"{'-'*50}")

    dcf_result = discounted_cash_flow_analysis(ticker)

    if dcf_result is None:
        print("❌ DCF analysis failed. Skipping sentiment adjustment.")
        return

    intrinsic_value, market_cap, upside, discounted_fcfs, discounted_terminal_value = dcf_result

    print(f"\n💰 Intrinsic Value: ${intrinsic_value:,.0f}")
    print(f"📈 Market Cap:      ${market_cap:,.0f}")
    print(f"📊 Upside Potential: {upside:.2f}%")

    news = get_security_news(ticker)
    sentiment_score = analyze_sentiment(news)
    adjusted_upside = upside + sentiment_score * 50

    print(f"\n🗞️ Sentiment Score (avg): {sentiment_score:.2f}")
    print(f"🎯 Adjusted Upside Potential: {adjusted_upside:.2f}%")

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(discounted_fcfs) + 1), discounted_fcfs, marker='o', label='Discounted FCF')
    plt.axhline(y=market_cap, color='r', linestyle='--', label='Market Cap')
    plt.title(f"{ticker} — 5-Year DCF Valuation")
    plt.xlabel("Year")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    ticker = input("🔎 Enter stock ticker (e.g., AAPL): ").upper().strip()
    evaluate_security(ticker)