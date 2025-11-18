import requests
import time
import json
import datetime
import os
from nsepython import nse_optionchain_scrapper

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"   # You can change the model

STATE_FILE = "state.json"

# ---------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------
def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing Telegram config")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram send error:", e)

# ---------------------------------------------------------
# PRICE FROM YAHOO
# ---------------------------------------------------------
def get_price_from_yahoo():
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1m"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except:
        return None

def get_price():
    print("Fetching Yahoo price...")
    p = get_price_from_yahoo()
    if p is not None:
        print("Fetched from Yahoo")
    return p

# ---------------------------------------------------------
# STATE HANDLING
# ---------------------------------------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_signal": None, "prices": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ---------------------------------------------------------
# SMA
# ---------------------------------------------------------
def calc_sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

# ---------------------------------------------------------
# MARKET TIME CHECK (IST)
# ---------------------------------------------------------
def is_market_time():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    weekday = now_utc.weekday()
    current_time = now_utc.time()

    if weekday >= 5:
        return False

    market_open = datetime.time(4, 0)   # 09:30 IST
    market_close = datetime.time(10, 0) # 15:30 IST

    return market_open <= current_time <= market_close

# ---------------------------------------------------------
# OPTION CHAIN USING NSEPYNTHON
# ---------------------------------------------------------
def get_option_chain():
    try:
        data = nse_optionchain_scrapper("NIFTY")
        return data
    except Exception as e:
        print("Option chain error:", e)
        return None

# ---------------------------------------------------------
# AI ANALYSIS USING OPENAI
# ---------------------------------------------------------
def ai_evaluate_signal(signal_type, price, sma9, sma21, option_chain):
    prompt = f"""
You are a trading decision AI.
You will receive:
1. BUY or SELL signal (from SMA cross)
2. Spot price
3. Short-term data (SMA9, SMA21)
4. Option Chain from NSE (full raw data)

Your task:
- Decide whether the signal is good or bad.
- Suggest the best strike price.
- Suggest whether CE or PE should be taken.
- Give a clear reason in simple words.

Input:
Signal: {signal_type}
Spot Price: {price}
SMA9: {sma9}
SMA21: {sma21}
Option Chain Data: {option_chain}

Return the response in plain text.
"""

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a financial analysis AI."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )

        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI error: {e}"

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
state = load_state()

while True:
    try:
        if not is_market_time():
            print("Market closed. Sleeping...")
            time.sleep(SLEEP_SECONDS)
            continue

        price = get_price()
        if price is None:
            print("No price available.")
            time.sleep(SLEEP_SECONDS)
            continue

        print("Price:", price)

        # Store price history
        state["prices"].append(price)
        if len(state["prices"]) > 50:
            state["prices"] = state["prices"][-50:]

        sma9 = calc_sma(state["prices"], 9)
        sma21 = calc_sma(state["prices"], 21)

        print(f"sma9 = {sma9}")
        print(f"sma21 = {sma21}")

        signal = None

        if sma9 is not None and sma21 is not None:
            if sma9 > sma21 and state["last_signal"] != "buy":
                signal = "BUY"
                state["last_signal"] = "buy"

            if sma9 < sma21 and state["last_signal"] != "sell":
                signal = "SELL"
                state["last_signal"] = "sell"

        # If signal generated → send to AI
        if signal:
            send_telegram(f"Signal Detected: {signal}\nPrice: {price}")

            option_chain = get_option_chain()

            ai_result = ai_evaluate_signal(signal, price, sma9, sma21, option_chain)
            send_telegram("AI Analysis:\n" + ai_result)

        save_state(state)

    except Exception as e:
        print("Error:", e)

    time.sleep(SLEEP_SECONDS)
