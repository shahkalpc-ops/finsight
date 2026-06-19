from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google import genai
from dotenv import load_dotenv
import os
import yfinance as yf
import ta
from database import init_db, save_prediction, get_predictions
import time
from google.genai import types

load_dotenv()
init_db()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_with_retry(prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            from google.genai import types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0)
            )
            return response.text
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f"Gemini error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/analyze.html", response_class=HTMLResponse)
def analyze_page():
    with open("templates/analyze.html") as f:
        return f.read()
    
@app.get("/watchlist.html", response_class=HTMLResponse)
def watchlist_page():
    with open("templates/watchlist.html") as f:
        return f.read()    
    
@app.get("/news/{ticker}")
def get_news(ticker: str):
    stock = yf.Ticker(ticker)
    news = stock.news

    if not news:
        return {"ticker": ticker, "articles": []}

    articles = []
    for item in news[:6]:
        title = item.get("content", {}).get("title", "")
        url = item.get("content", {}).get("canonicalUrl", {}).get("url", "")
        source = item.get("content", {}).get("provider", {}).get("displayName", "")
        
        if not title:
            continue

        articles.append({
            "title": title,
            "url": url,
            "source": source
        })

    if not articles:
        return {"ticker": ticker, "articles": []}

    headlines = "\n".join([f"- {a['title']}" for a in articles])
    prompt = f"""You are a financial analyst. For each headline below about {ticker} stock, classify it as BULLISH, BEARISH, or NEUTRAL for the stock price. 

Headlines:
{headlines}

Respond with exactly one word per headline in the same order: BULLISH, BEARISH, or NEUTRAL. One per line, nothing else."""

    try:
        sentiment_text = gemini_with_retry(prompt)
        sentiments = [s.strip().upper() for s in sentiment_text.strip().split("\n")]
    except:
        sentiments = ["NEUTRAL"] * len(articles)

    for i, article in enumerate(articles):
        article["sentiment"] = sentiments[i] if i < len(sentiments) else "NEUTRAL"

    return {"ticker": ticker, "articles": articles}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info

    name = info.get("longName", ticker)
    price = info.get("currentPrice", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    volume = info.get("volume", "N/A")
    market_cap = info.get("marketCap", "N/A")
    week_52_change = info.get("52WeekChange", "N/A")

    history = stock.history(period="1y")

    rsi = ta.momentum.RSIIndicator(history["Close"]).rsi().iloc[-1]
    macd = ta.trend.MACD(history["Close"])
    macd_value = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]
    ma50 = history["Close"].rolling(window=50).mean().iloc[-1]
    ma200 = history["Close"].rolling(window=200).mean().iloc[-1]

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

    analysis_text = gemini_with_retry(prompt)

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
        "analysis": analysis_text
    }

@app.post("/predict/{ticker}")
def generate_prediction(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info

    name = info.get("longName", ticker)
    price = info.get("currentPrice", "N/A")

    history = stock.history(period="1y")

    rsi = ta.momentum.RSIIndicator(history["Close"]).rsi().iloc[-1]
    macd = ta.trend.MACD(history["Close"])
    macd_value = macd.macd().iloc[-1]
    macd_signal_val = macd.macd_signal().iloc[-1]
    ma50 = history["Close"].rolling(window=50).mean().iloc[-1]
    ma200 = history["Close"].rolling(window=200).mean().iloc[-1]

    prompt = f"""
    You are a quantitative analyst. Based on these signals for {name} ({ticker}):
    - Current Price: ${price}
    - RSI: {rsi:.2f}
    - MACD: {macd_value:.4f} vs Signal: {macd_signal_val:.4f}
    - MA50: ${ma50:.2f}, MA200: ${ma200:.2f}

    Generate a structured prediction. Respond in exactly this format, no extra text:
    DIRECTION: [BULLISH or BEARISH or NEUTRAL]
    TARGET_PRICE: [number only]
    TIMEFRAME_DAYS: [number only, between 7 and 30]
    CONFIDENCE: [number only, between 30 and 90]
    REASONING: [2-3 sentences explaining the prediction based on the signals]
    """

    response_text = gemini_with_retry(prompt)

    lines = response_text.strip().split("\n")
    parsed = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip()] = val.strip()

    direction = parsed.get("DIRECTION", "NEUTRAL")
    target_price = float(parsed.get("TARGET_PRICE", price))
    timeframe_days = int(parsed.get("TIMEFRAME_DAYS", 14))
    confidence = int(parsed.get("CONFIDENCE", 50))
    reasoning = parsed.get("REASONING", "")

    save_prediction(ticker, name, direction, price, target_price, confidence, timeframe_days, reasoning)

    return {
        "ticker": ticker,
        "name": name,
        "current_price": price,
        "direction": direction,
        "target_price": target_price,
        "timeframe_days": timeframe_days,
        "confidence": confidence,
        "reasoning": reasoning
    }

@app.get("/watchlist")
def get_watchlist():
    rows = get_predictions()
    predictions = []
    for row in rows:
        predictions.append({
            "id": row[0],
            "ticker": row[1],
            "company_name": row[2],
            "direction": row[3],
            "price_at_prediction": row[4],
            "target_price": row[5],
            "confidence": row[6],
            "timeframe_days": row[7],
            "reasoning": row[8],
            "created_at": row[9],
            "resolved": row[10],
            "outcome": row[11]
        })
    return {"predictions": predictions}