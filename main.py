from fastapi import FastAPI
from google import genai
from dotenv import load_dotenv
import os
import yfinance as yf
import ta
import pandas as pd

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FinSight is alive"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Basic info
    name = info.get("longName", ticker)
    price = info.get("currentPrice", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    volume = info.get("volume", "N/A")
    market_cap = info.get("marketCap", "N/A")
    week_52_change = info.get("52WeekChange", "N/A")

    # Get 6 months of daily price history
    history = stock.history(period="1y")

    # Calculate technical indicators
    rsi = ta.momentum.RSIIndicator(history["Close"]).rsi().iloc[-1]
    macd = ta.trend.MACD(history["Close"])
    macd_value = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]
    ma50 = history["Close"].rolling(window=50).mean().iloc[-1]
    ma200 = history["Close"].rolling(window=200).mean().iloc[-1]

    # Determine signals
    rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    macd_signal_text = "bullish" if macd_value > macd_signal else "bearish"
    ma_signal = "bullish" if price > ma200 else "bearish"

    prompt = f"""
    Analyze {name} ({ticker}) stock given these real current metrics:
    - Current Price: ${price}
    - 52 Week Change: {week_52_change}
    - P/E Ratio: {pe_ratio}
    - Volume: {volume}
    - Market Cap: {market_cap}

    Technical Indicators:
    - RSI: {rsi:.2f} ({rsi_signal})
    - MACD: {macd_value:.4f} vs Signal: {macd_signal:.4f} ({macd_signal_text})
    - 50 Day MA: ${ma50:.2f}
    - 200 Day MA: ${ma200:.2f} ({ma_signal} trend)

    Give a concise analysis covering: what these signals mean together, 
    key risks, and an overall bullish/bearish/neutral rating with confidence level.
    Be specific to these exact numbers.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "ticker": ticker,
        "name": name,
        "price": price,
        "pe_ratio": pe_ratio,
        "volume": volume,
        "market_cap": market_cap,
        "week_52_change": week_52_change,
        "technicals": {
            "rsi": round(rsi, 2),
            "rsi_signal": rsi_signal,
            "macd": round(macd_value, 4),
            "macd_signal": macd_signal_text,
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "ma_trend": ma_signal
        },
        "analysis": response.text
    }