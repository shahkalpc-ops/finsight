from fastapi import FastAPI
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FinSight is alive"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    prompt = f"Give me a brief investment analysis of {ticker} stock. Cover recent performance, key risks, and whether the signals look bullish or bearish right now."
    response = model.generate_content(prompt)
    return {"ticker": ticker, "analysis": response.text}