import requests
import time
import json
import datetime
import os
from nsepython import nse_optionchain_scrapper
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any, List, Optional
# The 'datetime' import is correctly available from the previous imports.

# ---------------------------------------------------------
## ⚙️ CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Gemini API is used for AI Analysis
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash" 

STATE_FILE = "state.json"

try:
    # Initialize the client globally
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Gemini client: {e}")
    # If the client can't initialize, we exit as the core feature won't work.
    exit(1) # Use exit(1) to signal an error state

# ---------------------------------------------------------
## 💬 TELEGRAM
# ---------------------------------------------------------
def send_telegram(msg: str):
    """Sends a message to the configured Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        # Using print for debugging if Telegram is not configured
        print("⚠️ Missing Telegram config (BOT_TOKEN or CHAT_ID). Message not sent.")
        print(f"TELEGRAM MESSAGE: {msg}")
        return
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        # Use Markdown parsing mode for better message formatting
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except Exception as e:
        print("❌ Telegram send error:", e)

# ---------------------------------------------------------
## 💰 PRICE FETCHING & STATE
# ---------------------------------------------------------
def get_price_from_yahoo() -> Optional[float]:
    """Fetches the Nifty spot price from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1m"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception as e:
        print(f"❌ Error fetching price from Yahoo: {e}")
        return None

def get_price() -> Optional[float]:
    """Retrieves the price, logging the source."""
    p = get_price_from_yahoo()
    if p is not None:
        print(f"✅ Fetched price: {p}")
    return p

def load_state() -> Dict[str, Any]:
    """Loads price history and last signal from state file."""
    if not os.path.exists(STATE_FILE):
        return {"last_signal": None, "prices": []}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading state file: {e}")
        return {"last_signal": None, "prices": []}

def save_state(state: Dict[str, Any]):
    """Saves price history and last signal to state file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ---------------------------------------------------------
## 📊 SMA & MARKET CHECK
# ---------------------------------------------------------
def calc_sma(values: List[float], period: int) -> Optional[float]:
    """Calculates the Simple Moving Average (SMA)."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def is_market_time() -> bool:
    """Checks if the current time (converted to IST) is within Indian market hours (Mon-Fri, 9:15 AM - 3:30 PM IST)."""
    # NOTE: This uses a simplified UTC check (4:00 to 10:00 UTC) which corresponds to 9:30 AM to 3:30 PM IST.
    # The market actually opens at 9:15 AM IST (3:45 UTC), but 9:30 AM (4:00 UTC) is close enough for a simple script.
    
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    weekday = now_utc.weekday()
    
    # Check for Saturday (5) and Sunday (6)
    if weekday >= 5:
        return False

    # Define time boundaries in UTC
    market_open_utc = datetime.time(3, 45)  # 09:15 IST
    market_close_utc = datetime.time(10, 0) # 15:30 IST

    return market_open_utc <= now_utc.time() <= market_close_utc

# ---------------------------------------------------------
## 📈 OPTION CHAIN (NSEPYNTHON)
# ---------------------------------------------------------
def get_nifty_strikes_for_expiry() -> Optional[Dict[str, Any]]:
    """
    Fetches NIFTY options data, calculates ATM, and filters strikes around ATM 
    for the closest available expiry date.
    """
    try:
        # Fetch the full option chain
        data = nse_optionchain_scrapper("NIFTY")
        
        spot_price = data['records']['underlyingValue']
        option_data = data['records']['data']

        # Get the closest expiry date directly from the fetched data
        expiry_dates = data['records']['expiryDates']
        if not expiry_dates:
            print("❌ Error: No expiry dates found in the NSE data.")
            return None
            
        expiry = expiry_dates[0] 
        print(f"✅ Using closest expiry: {expiry}")

        # Calculate ATM strike (nearest 50 for NIFTY)
        atm = round(spot_price / 50) * 50

        # ATM ±3 strikes (step 50). This results in a total of 7 strikes.
        strikes_needed = [atm + i*50 for i in range(-3, 4)] 

        # Filter only required strikes for the closest expiry
        filtered_records = [
            record for record in option_data 
            if record['strikePrice'] in strikes_needed and record['expiryDate'] == expiry
        ]

        return {
            "spot": spot_price,
            "atm": atm,
            "required_strikes": strikes_needed,
            "expiry": expiry,
            "records": filtered_records # This is the list the AI function needs
        }
    except Exception as e:
        print(f"❌ Error fetching NSE data: {e}")
        return None

# ---------------------------------------------------------
## 🤖 AI ANALYSIS (GEMINI)
# ---------------------------------------------------------
def prepare_gemini_prompt(strike_data: List[Dict[str, Any]]) -> str:
    """Converts filtered strike data into a focused JSON string for the prompt."""
    return json.dumps(strike_data, indent=2)

def get_ai_trade_suggestion(option_chain_data: List[Dict[str, Any]], price: float, sma9: float, sma21: float, signal_type: str) -> str:
    """
    Evaluates a trading signal and Nifty Options Chain using the Gemini API.
    """
    if not client:
        return "AI error: Gemini client is not initialized. Check API Key."

    option_chain_str = prepare_gemini_prompt(option_chain_data)

    user_prompt = f"""
Input:
Signal: {signal_type}
Spot Price: {price}
SMA9: {sma9}
SMA21: {sma21}
Option Chain Data (Filtered JSON):
{option_chain_str}

Your task:
- Analyze the SMA signal (trend confirmation) and the Options Chain (support/resistance/sentiment).
- Determine a **Confidence Level** (Low, Medium, or High) for the overall trading decision based on whether the Options Chain confirms or contradicts the SMA signal.
- Decide whether the SMA cross signal is **Good** (confirmed by options data) or **Bad** (contradicted by options data).
- Suggest the best **Strike Price** for the trade.
- Suggest whether a **CE** (Call Option) or **PE** (Put Option) should be taken.
- Give a **clear, concise reason** in simple, layman words, specifically mentioning key OI levels.

Return the response in **plain text** only. The response MUST start with the Confidence Level, followed by a colon and a space, then the rest of the analysis.

Example desired format:
Confidence: High. Signal is Good. Strike Price: 25000. Option: CE. Reason: The Options Chain shows strong PE OI building at 25000, confirming the bullish SMA cross.
"""
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": "You are a highly experienced NIFTY options trading decision AI. Your goal is to combine technical and options data for actionable, risk-aware advice."}]},
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config=genai.types.GenerateContentConfig(temperature=0.2)
        )

        return response.text
    
    except APIError as e:
        return f"AI API Error: {e}. Check API Key or rate limits."
    except Exception as e:
        return f"Unexpected AI error: {e}"

# ---------------------------------------------------------
## 🏃 MAIN EXECUTION LOOP
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
            print("No price available, skipping iteration.")
            time.sleep(SLEEP_SECONDS)
            continue

        # 1. Update Price History
        state["prices"].append(price)
        # Keep only the last 50 prices for SMA calculation
        if len(state["prices"]) > 50:
            state["prices"] = state["prices"][-50:]

        # 2. Calculate SMAs
        sma9 = calc_sma(state["prices"], 9)
        sma21 = calc_sma(state["prices"], 21)

        print(f"SMA9: {sma9} | SMA21: {sma21}")

        signal = None

        # 3. Check for SMA Crossover Signal
        if sma9 is not None and sma21 is not None:
            # BUY signal: SMA9 crosses ABOVE SMA21
            if sma9 > sma21 and state["last_signal"] != "buy":
                signal = "BUY"
                state["last_signal"] = "buy"

            # SELL signal: SMA9 crosses BELOW SMA21
            elif sma9 < sma21 and state["last_signal"] != "sell":
                signal = "SELL"
                state["last_signal"] = "sell"

        # 4. If Signal Generated, Get AI Analysis and Notify
        if signal:
            send_telegram(f"*🚨 Signal Detected: {signal}* (Price: {price:.2f})")

            # Fetch and filter Options Chain data
            option_chain_result = get_nifty_strikes_for_expiry()
            
            if option_chain_result and option_chain_result['records']:
                # CRITICAL FIX: Pass the 'records' list and the SMA values
                ai_result = get_ai_trade_suggestion(
                    option_chain_data=option_chain_result['records'], 
                    price=price, 
                    sma9=sma9, 
                    sma21=sma21, 
                    signal_type=signal
                )
                send_telegram("*🤖 AI Analysis:*\n" + ai_result)
            else:
                send_telegram(f"*⚠️ Warning:* Failed to fetch valid Options Chain data for {signal} signal.")

        # 5. Save State
        save_state(state)

    except Exception as e:
        print(f"🔥 UNHANDLED ERROR IN MAIN LOOP: {e}")
        send_telegram(f"*FATAL ERROR in Trading Bot:*\n{e}")

    # Sleep until the next iteration
    time.sleep(SLEEP_SECONDS)