import requests
import time
import json
import datetime
import os
import logging
import sys
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type 
from nsepython import nse_optionchain_scrapper
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------
## ⚙️ LOGGING SETUP (Console Only)
# ---------------------------------------------------------
# Configure logging to output only to the console (sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
## ⚙️ CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
try:
    SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))
except ValueError:
    SLEEP_SECONDS = 60
    logger.warning("SLEEP_SECONDS environment variable is invalid. Defaulting to 60 seconds.")


# Gemini API is used for AI Analysis
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash" 

STATE_FILE = "state.json"
MARKET_CLOSE_HOUR_UTC = 10 # 3:30 PM IST is 10:00 AM UTC
MARKET_CLOSE_MINUTE_UTC = 0

try:
    # Initialize the client globally
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini client initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing Gemini client: {e}. Exiting script.")
    exit(1)

# ---------------------------------------------------------
## 💬 TELEGRAM & STATE UTILITIES (Functions remain the same)
# ---------------------------------------------------------

def send_telegram(msg: str):
    """Sends a message to the configured Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning(f"⚠️ Missing Telegram config. Content: {msg.strip().replace('\n', ' ')}")
        return
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        logger.info("Telegram message sent successfully.")
    except Exception as e:
        logger.error(f"❌ Telegram send error: {e}")

def get_price_from_yahoo() -> Optional[float]:
    """Fetches the Nifty spot price from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1m"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception as e:
        logger.error(f"❌ Error fetching price from Yahoo: {e}")
        return None

def get_price() -> Optional[float]:
    """Retrieves the price, logging the source."""
    p = get_price_from_yahoo()
    if p is not None:
        logger.info(f"✅ Fetched price: {p:.2f}")
    return p

def load_state() -> Dict[str, Any]:
    """Loads price history and last signal from state file."""
    if not os.path.exists(STATE_FILE):
        logger.info("State file not found. Initializing new state.")
        return {"last_signal": None, "prices": [], "last_state_clear_date": None}
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # Ensure new keys are present even if file is old
            state.setdefault("last_state_clear_date", None)
            logger.info(f"State loaded successfully. Last signal: {state.get('last_signal')}")
            return state
    except Exception as e:
        logger.error(f"❌ Error loading state file, resetting state: {e}")
        return {"last_signal": None, "prices": [], "last_state_clear_date": None}

def save_state(state: Dict[str, Any]):
    """Saves price history and last signal to state file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        logger.debug("State saved successfully.")
    except Exception as e:
        logger.error(f"❌ Error saving state file: {e}")

def calc_sma(values: List[float], period: int) -> Optional[float]:
    """Calculates the Simple Moving Average (SMA)."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def is_market_time() -> bool:
    """
    Checks if the current time is within Indian market hours (Mon-Fri, 9:15 AM - 3:30 PM IST).
    9:15 AM IST = 3:45 AM UTC
    3:30 PM IST = 10:00 AM UTC
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    weekday = now_utc.weekday()
    if weekday >= 5: # Saturday or Sunday
        return False
        
    # 9:15 AM IST is 03:45 UTC
    market_open_utc = datetime.time(3, 45) 
    # 3:30 PM IST is 10:00 UTC
    market_close_utc = datetime.time(MARKET_CLOSE_HOUR_UTC, MARKET_CLOSE_MINUTE_UTC)
    
    return market_open_utc <= now_utc.time() <= market_close_utc
    
def check_and_clear_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks if the market is closed and if the state file needs to be cleared for a new day.
    This ensures we start every trading day with a fresh state (SMA and last_signal).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    market_close_time_utc = datetime.time(MARKET_CLOSE_HOUR_UTC, MARKET_CLOSE_MINUTE_UTC)
    today_date_str = now_utc.date().isoformat()
    
    # 1. Check if it's past market close
    if now_utc.time() > market_close_time_utc and today_date_str != state.get("last_state_clear_date"):
        logger.critical(f"📊 Market is closed. Clearing state file for a fresh start tomorrow.")
        
        # Clear the STATE_FILE
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logger.info(f"✅ State file '{STATE_FILE}' deleted.")
        
        # Return a new, clean state and update the last clear date
        new_state = {"last_signal": None, "prices": [], "last_state_clear_date": today_date_str}
        save_state(new_state)
        return new_state
        
    return state
    
# ... (get_nifty_strikes_for_expiry and prepare_gemini_prompt remain the same) ...

def get_nifty_strikes_for_expiry() -> Optional[Dict[str, Any]]:
    """Fetches NIFTY options data and filters strikes around ATM."""
    try:
        # NSE data fetching logic... (unchanged)
        data = nse_optionchain_scrapper("NIFTY")
        spot_price = data['records']['underlyingValue']
        option_data = data['records']['data']
        expiry_dates = data['records']['expiryDates']
        if not expiry_dates:
            logger.warning("❌ No expiry dates found in the NSE data.")
            return None
            
        expiry = expiry_dates[0] 
        atm = round(spot_price / 50) * 50
        strikes_needed = [atm + i*50 for i in range(-3, 4)] 

        filtered_records = [
            record for record in option_data 
            if record['strikePrice'] in strikes_needed and record['expiryDate'] == expiry
        ]
        
        logger.info(f"Filtered {len(filtered_records)} option records around ATM {atm}.")

        return {
            "spot": spot_price,
            "atm": atm,
            "expiry": expiry,
            "records": filtered_records
        }
    except Exception as e:
        logger.error(f"❌ Error fetching or processing NSE data: {e}")
        return None

def prepare_gemini_prompt(strike_data: List[Dict[str, Any]]) -> str:
    """Converts filtered strike data into a focused JSON string for the prompt."""
    return json.dumps(strike_data, indent=2)

# ---------------------------------------------------------
## 🤖 AI ANALYSIS (GEMINI) - WITH TENACITY BACKOFF
# ---------------------------------------------------------
# ... (_call_gemini_with_retry remains the same)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), 
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError)
)
def _call_gemini_with_retry(client, model, contents, config):
    """
    Internal function to call the Gemini API, wrapped with the tenacity retry logic.
    """
    logger.debug("Executing Gemini API call...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    logger.debug("Gemini call completed successfully.")
    return response.text


def get_ai_trade_suggestion(option_chain_data: List[Dict[str, Any]], price: float, sma9: float, sma21: float, signal_type: str) -> str:
    """
    Evaluates a trading signal and Nifty Options Chain using the Gemini API, 
    managing retries for transient errors and requesting TP/SL targets.
    """
    if not client:
        return "AI error: Gemini client is not initialized."

    option_chain_str = prepare_gemini_prompt(option_chain_data)

    # --- UPDATED PROMPT: Added Take Profit (TP) and Stop Loss (SL) requirements ---
    user_prompt = f"""
Input:
Signal: {signal_type}
Spot Price: {price}
SMA9: {sma9:.2f}
SMA21: {sma21:.2f}
Option Chain Data (Filtered JSON):
{option_chain_str}

Your task:
- Analyze the SMA signal and the Options Chain (support/resistance/sentiment).
- Determine a **Confidence Level** (Low, Medium, or High).
- Decide whether the SMA cross signal is **Good** (confirmed) or **Bad** (contradicted).
- Suggest the best **Strike Price** and **Option** (CE/PE).
- Based on the Options Chain's **Open Interest (OI)** and **Change in OI (Chg in OI)**, identify the **nearest strong resistance/support** level that should serve as a **Take Profit (TP) Target** (NIFTY price level).
- Based on the Options Chain's OI/Chg in OI, identify the **nearest strong support/resistance** level on the opposite side that should serve as a **Stop Loss (SL) Target** (NIFTY price level).
- Give a **clear, concise reason** in simple, layman words, specifically mentioning key OI levels.

Return the response in **plain text** only. The response MUST include the TP and SL levels in the specified format.

Example desired format:
Confidence: High. Signal: Good. Strike Price: 25000. Option: CE. Take Profit (TP): 25150. Stop Loss (SL): 24900. Reason: Strong PE OI at 25000 confirms the bullish SMA cross.
"""

    try:
        logger.info("Starting Gemini API call (up to 5 attempts with backoff)...")
        response_text = _call_gemini_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": "You are a highly experienced NIFTY options trading decision AI. Your goal is to combine technical and options data for actionable, risk-aware advice."}]},
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config=genai.types.GenerateContentConfig(temperature=0.2)
        )
        return response_text
        
    except APIError as e:
        final_message = f"AI API Error: Failed after 5 retries. The model may be overloaded (503), or check your quota (429). Details: {e}"
        logger.error(final_message)
        return final_message
        
    except Exception as e:
        logger.error(f"❌ Unexpected non-API error in AI suggestion: {e}")
        return f"Unexpected AI error: {e}"


# ---------------------------------------------------------
## 🏃 MAIN EXECUTION LOOP
# ---------------------------------------------------------
state = load_state()
logger.info("Starting Main Trading Bot Loop.")

while True:
    try:
        # Check and clear state if market is closed (runs once after market close)
        state = check_and_clear_state(state)
        
        if not is_market_time():
            logger.info("Market closed or weekend. Sleeping...")
            time.sleep(SLEEP_SECONDS)
            continue

        price = get_price()
        if price is None:
            logger.warning("No price available, skipping iteration.")
            time.sleep(SLEEP_SECONDS)
            continue

        # 1. Update Price History
        state["prices"].append(price)
        if len(state["prices"]) > 50:
            state["prices"] = state["prices"][-50:]

        # 2. Calculate SMAs
        sma9 = calc_sma(state["prices"], 9)
        sma21 = calc_sma(state["prices"], 21)

        if sma9 is None or sma21 is None:
            logger.info(f"Insufficient data ({len(state['prices'])} points) for full SMA calculation. Sleeping.")
            save_state(state)
            time.sleep(SLEEP_SECONDS)
            continue

        logger.info(f"Current Price: {price:.2f} | SMA9: {sma9:.2f} | SMA21: {sma21:.2f}")

        signal = None
        current_trend = "buy" if sma9 > sma21 else "sell" if sma9 < sma21 else "neutral"

        # 3. Check for SMA Crossover Signal (Signal persistence logic)
        if current_trend == "buy" and state["last_signal"] != "buy":
            signal = "BUY"
            state["last_signal"] = "buy"

        elif current_trend == "sell" and state["last_signal"] != "sell":
            signal = "SELL"
            state["last_signal"] = "sell"

        # 4. If Signal Generated, Get AI Analysis and Notify
        if signal:
            logger.critical(f"🚨 MAJOR SIGNAL DETECTED: {signal} at Price {price:.2f}")
            send_telegram(f"*🚨 Major Signal Detected: {signal}* (Price: {price:.2f})")

            option_chain_result = get_nifty_strikes_for_expiry()
            
            if option_chain_result and option_chain_result['records']:
                ai_result = get_ai_trade_suggestion(
                    option_chain_data=option_chain_result['records'], 
                    price=price, 
                    sma9=sma9, 
                    sma21=sma21, 
                    signal_type=signal
                )
                ai_log_message = ai_result.strip().replace('\n', ' | ')
                logger.critical(f"🤖 AI RECOMMENDS: {ai_log_message}")
                send_telegram("*🤖 AI Analysis:*\n" + ai_result)
            else:
                logger.warning(f"⚠️ Failed to fetch valid Options Chain for {signal} signal. Skipping AI analysis.")
                send_telegram(f"*⚠️ Warning:* Failed to fetch valid Options Chain data for {signal} signal.")

        # 5. Save State
        save_state(state)

    except Exception as e:
        logger.exception(f"🔥 UNHANDLED ERROR IN MAIN LOOP: {e}")
        send_telegram(f"*🔥 FATAL ERROR in Trading Bot:*\n`{e}`")

    # Sleep until the next iteration
    time.sleep(SLEEP_SECONDS)