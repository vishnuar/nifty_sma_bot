import requests
import time
import json
import datetime
import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

STATE_FILE = "state.json"

def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing Telegram config")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

def get_price_from_nse():
    try:
        url = "https://www.nseindia.com/api/quote-equity?symbol=NIFTY 50"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers)
        r = s.get(url, headers=headers)
        data = r.json()
        return float(data["priceInfo"]["lastPrice"])
    except:
        return None

def get_price_from_yahoo():
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1m"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except:
        return None

def get_price():
    p = get_price_from_nse()
    if p is not None:
        print("Fetched from NSE")
        return p
    print("NSE failed… trying Yahoo")
    p = get_price_from_yahoo()
    if p is not None:
        print("Fetched from Yahoo")
    return p

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_signal": None, "prices": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def calc_sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

# ========================================================
# FIXED MARKET TIME FUNCTION (UTC → IST conversion)
# ========================================================
def is_market_time():
    now_utc = datetime.datetime.now(datetime.timezone.utc)   # <<< changed
    weekday = now_utc.weekday()                              # <<< changed
    current_time = now_utc.time()                            # <<< changed

    if weekday >= 5:
        return False

    # IST 09:30–15:30 → UTC 04:00–10:00
    market_open = datetime.time(4, 0)   # <<< changed
    market_close = datetime.time(10, 0) # <<< changed

    return market_open <= current_time <= market_close        # <<< changed

# ========================================================
# MAIN LOOP
# ========================================================
state = load_state()

while True:
    try:
        if not is_market_time():
            print("Market closed. Sleeping...")
            time.sleep(SLEEP_SECONDS)
            continue

        price = get_price()
        if price is None:
            print("No price source available.")
            time.sleep(SLEEP_SECONDS)
            continue

        state["prices"].append(price)
        if len(state["prices"]) > 50:
            state["prices"] = state["prices"][-50:]

        sma9 = calc_sma(state["prices"], 9)
        sma21 = calc_sma(state["prices"], 21)

        print("sma9" + sma9)
        print("sma21" + sma21)

        if sma9 and sma21:
            if sma9 > sma21 and state["last_signal"] != "buy":
                send_telegram(f"BUY Signal — SMA9 crossed above SMA21\nPrice: {price}")
                state["last_signal"] = "buy"

            if sma9 < sma21 and state["last_signal"] != "sell":
                send_telegram(f"SELL Signal — SMA9 crossed below SMA21\nPrice: {price}")
                state["last_signal"] = "sell"

        save_state(state)

    except Exception as e:
        print("Error:", e)

    time.sleep(SLEEP_SECONDS)
