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
    MARKET_SLEEP_SECONDS = int(os.getenv("MARKET_SLEEP_SECONDS", "300"))
except ValueError:
    MARKET_SLEEP_SECONDS = 300
    logger.warning("MARKET_SLEEP_SECONDS environment variable is invalid. Defaulting to 300 seconds.")

try:
    PRICE_FETCH_DELAY = int(os.getenv("PRICE_FETCH_DELAY", "60"))
except ValueError:
    PRICE_FETCH_DELAY = 60
    logger.warning("PRICE_FETCH_DELAY environment variable is invalid. Defaulting to 60 seconds.")


# Gemini API is used for AI Analysis
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash" 

STATE_FILE = "state.json"
MARKET_CLOSE_HOUR_UTC = 10 # 3:30 PM IST is 10:00 AM UTC
MARKET_CLOSE_MINUTE_UTC = 0

# --- NEW CONFIGURATION: SMA Buffer (Read from environment) ---
try:
    SMA_BUFFER_POINTS = float(os.getenv("SMA_BUFFER", "3.0"))
except ValueError:
    SMA_BUFFER_POINTS = 3.0
    logger.warning("SMA_BUFFER environment variable is invalid. Defaulting to 3.0 points.")
# --- END NEW CONFIGURATION ---

try:
    # Initialize the client globally
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini client initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing Gemini client: {e}. Exiting script.")
    exit(1)

# ---------------------------------------------------------
## 💬 TELEGRAM & STATE UTILITIES
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

def calculate_max_pain_and_pcr(option_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Max Pain and Put-Call Ratio (PCR) from the full option chain data."""
    if not option_data:
        return {"max_pain": "N/A", "pcr": "N/A"}

    # 1. Calculate PCR
    total_put_oi = 0
    total_call_oi = 0
    for record in option_data:
        if record.get('PE') and isinstance(record['PE'].get('openInterest'), (int, float)):
            total_put_oi += record['PE']['openInterest']
        if record.get('CE') and isinstance(record['CE'].get('openInterest'), (int, float)):
            total_call_oi += record['CE']['openInterest']

    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0.0

    # 2. Calculate Max Pain
    max_pain = "N/A"
    min_loss = float('inf')

    # We use all strike prices from the input data, not just the filtered ones
    strike_prices = sorted(list(set(r['strikePrice'] for r in option_data)))

    for strike in strike_prices:
        total_loss_at_strike = 0

        for record in option_data:
            record_strike = record['strikePrice']

            # Loss for Put Writers (PE is ITM, PE Writers lose)
            if record_strike < strike and record.get('PE') and isinstance(record['PE'].get('openInterest'), (int, float)):
                total_loss_at_strike += (strike - record_strike) * record['PE']['openInterest']

            # Loss for Call Writers (CE is ITM, CE Writers lose)
            if record_strike > strike and record.get('CE') and isinstance(record['CE'].get('openInterest'), (int, float)):
                total_loss_at_strike += (record_strike - strike) * record['CE']['openInterest']

        if total_loss_at_strike < min_loss:
            min_loss = total_loss_at_strike
            max_pain = strike

    return {"max_pain": max_pain, "pcr": pcr}

def get_nifty_strikes_for_expiry() -> Optional[Dict[str, Any]]:
    """Fetches NIFTY options data, calculates Max Pain/PCR, and filters strikes around ATM."""
    try:
        data = nse_optionchain_scrapper("NIFTY")
        spot_price = data['records']['underlyingValue']
        full_option_data = data['records']['data']
        expiry_dates = data['records']['expiryDates']
        if not expiry_dates:
            logger.warning("❌ No expiry dates found in the NSE data.")
            return None

        expiry = expiry_dates[0] 
        atm = round(spot_price / 50) * 50
        strikes_needed = [atm + i*50 for i in range(-7, 8)] 

        # Filter records for AI analysis (around ATM)
        filtered_records = [
            record for record in full_option_data 
            if record['strikePrice'] in strikes_needed and record['expiryDate'] == expiry
        ]

        # Calculate Max Pain and PCR using the FULL option chain data
        metrics = calculate_max_pain_and_pcr(full_option_data)

        logger.info(f"Filtered {len(filtered_records)} option records. Max Pain: {metrics['max_pain']}, PCR: {metrics['pcr']:.2f}")

        return {
            "spot": spot_price,
            "atm": atm,
            "expiry": expiry,
            "records": filtered_records,
            "pcr": metrics['pcr'],
            "max_pain": metrics['max_pain']
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


def get_ai_trade_suggestion(option_chain_data: List[Dict[str, Any]], price: float, sma9: float, sma21: float, signal_type: str, pcr: float, max_pain: str) -> str:
    """
    Evaluates a trading signal and Nifty Options Chain using the Gemini API.
    """
    if not client:
        return "AI error: Gemini client is not initialized."

    option_chain_str = prepare_gemini_prompt(option_chain_data)

    # Extract the expiry date safely from the filtered records list
    expiry_date = option_chain_data[0].get('expiryDate', 'N/A') if option_chain_data else 'N/A'

    # --- REVISED PROMPT WITH PCR, MAX PAIN, and NEW WRITING LOGIC ---
    user_prompt = f"""
**SYSTEM PROMPT: You are a highly specialized and experienced NIFTY options market analyst and strategist. Your sole function is to combine the provided technical (SMA) signal with Open Interest (OI) data, PCR, and Max Pain to generate a single, actionable, risk-managed trading recommendation.**

Input Data:
Signal: {signal_type}
Spot Price: {price:.2f}
SMA9: {sma9:.2f}
SMA21: {sma21:.2f}
Current UTC Date: {datetime.datetime.now(datetime.timezone.utc).date().isoformat()}
Option Expiry Date: {expiry_date}
Put-Call Ratio (PCR): {pcr:.2f}
Max Pain Level: {max_pain}
Option Chain Data (Filtered JSON):
{option_chain_str}

--- GUIDELINES AND CONSTRAINTS ---

1.  **Definitions & Data Constraint:**
    * **Resistance (TP Target):** Strong Call Option (CE) Open Interest (OI) or Change in OI build-up.
    * **Support (SL Target for BUY/TP Target for SELL):** Strong Put Option (PE) Open Interest (OI) or Change in OI build-up.
    * **Strike Price** and **NIFTY Price Levels (TP/SL)** MUST be selected ONLY from the strike prices provided in the 'Option Chain Data' JSON. DO NOT create a numerical value that is not present.
2.  **Trade Parameters:**
    * **Take Profit (TP) Target** MUST be set at the nearest strong **Resistance (CE OI)** level.
    * **Stop Loss (SL) Target** MUST be set at the nearest strong **Conflicting New Writing** level.

3.  **MARKET STRUCTURE (PCR/MAX PAIN) ANALYSIS:**
    * **New Writing (Conviction):** The AI must prioritize signals confirmed by new writing over other OI metrics.

4.  **VOLATILITY AND EXPIRY DAY RULE:**
    * **If today's date matches the Option Expiry Date ({expiry_date}), the market is highly volatile.** Automatically apply a one-tier downgrade to the initial **Confidence Level** (e.g., Very High -> High, High -> Medium, Medium -> Low).

5.  **Confidence & R/R Constraint (R/R > 1.5):**
    * **Confidence Level** can be: **(Very High, High, Medium, or Low).**
    * If the calculated Risk/Reward (R/R) ratio is less than 1.5, the final confidence MUST be **Low**.

--- REQUIRED OUTPUT FORMAT ---

**Output MUST be a single, continuous line of plain text and layman words**
**Output MUST contain ALL of the following key-value pairs in the exact order shown below.**
**The Reason MUST be a single, concise sentence that justifies the decision by referencing the SMA, PCR, and the key OI levels used for TP/SL.**

Example desired format:
Confidence: High. Signal: Buy. Strike Price: 25000. Option: CE. Take Profit (TP): 25150. Stop Loss (SL): 24900. Reason: SMA confirms signal, PCR 1.12 supports rally, and new PE writing at 25000 confirms conviction.
"""

    try:
        logger.info("Starting Gemini API call (up to 5 attempts with backoff)...")
        response_text = _call_gemini_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": "You are a highly experienced NIFTY options trading decision AI. Your goal is to combine technical, PCR, Max Pain, and options data for actionable, risk-aware advice."}]},
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

        if not is_market_time():
            logger.info("Market closed or weekend. Sleeping...")
            time.sleep(MARKET_SLEEP_SECONDS)
            continue

        price = get_price()
        if price is None:
            logger.warning("No price available, skipping iteration.")
            time.sleep(PRICE_FETCH_DELAY)
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
            time.sleep(PRICE_FETCH_DELAY)
            continue

        logger.info(f"Current Price: {price:.2f} | SMA9: {sma9:.2f} | SMA21: {sma21:.2f}")

        signal = None
        SMA_BUFFER = SMA_BUFFER_POINTS # Use the configuration constant 

        if sma9 > (sma21 + SMA_BUFFER):
            current_trend = "buy"  # SMA9 must be 3 points ABOVE SMA21
        elif sma9 < (sma21 - SMA_BUFFER):
            current_trend = "sell" # SMA9 must be 3 points BELOW SMA21
        else:
            logger.info("Neutral trend detected. Keep contuning the same trend.")
            time.sleep(PRICE_FETCH_DELAY)
            continue

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
                # Call AI with new PCR and Max Pain data
                ai_result = get_ai_trade_suggestion(
                    option_chain_data=option_chain_result['records'], 
                    price=price, 
                    sma9=sma9, 
                    sma21=sma21, 
                    signal_type=signal,
                    pcr=option_chain_result['pcr'],
                    max_pain=str(option_chain_result['max_pain']) # Pass as string for safety
                )
                ai_log_message = ai_result.strip().replace('\n', ' | ')
                logger.critical(f"🤖 AI RECOMMENDS: {ai_log_message}")
                send_telegram("*🤖 AI Analysis:*\n" + ai_result)
            else:
                logger.warning(f"⚠️ Failed to fetch valid Options Chain for {signal} signal. Skipping AI analysis.")
                send_telegram(f"*⚠️ Warning:* Failed to fetch valid Options Chain data for {signal} signal.")

        # 5. Save State
        save_state(state)
        time.sleep(PRICE_FETCH_DELAY)
        

    except Exception as e:
        logger.exception(f"🔥 UNHANDLED ERROR IN MAIN LOOP: {e}")
        send_telegram(f"*🔥 FATAL ERROR in Trading Bot:*\n`{e}`")

    # Sleep until the next iteration