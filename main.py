from fastapi import FastAPI
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import yfinance as yf

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
    
    name = info.get("longName", ticker)
    price = info.get("currentPrice", "N/A")
    change = info.get("52WeekChange", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    volume = info.get("volume", "N/A")
    market_cap = info.get("marketCap", "N/A")

    prompt = f"""
    Analyze {name} ({ticker}) stock given these real current metrics:
    - Current Price: ${price}
    - 52 Week Change: {change}
    - P/E Ratio: {pe_ratio}
    - Volume: {volume}
    - Market Cap: {market_cap}

    Give a concise analysis covering: what these numbers mean, key risks, 
    and whether signals look bullish or bearish. Be specific to these numbers.
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
        "analysis": response.text
    }