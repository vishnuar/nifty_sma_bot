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
    """Loads price history, signal data, and feedback status from state file."""
    if not os.path.exists(STATE_FILE):
        logger.info("State file not found. Initializing new state.")
        # Removed time-based keys
        return {"last_signal": None, "prices": [], "last_state_clear_date": None, "last_trade_feedback": "NONE"}
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # Ensure new keys are present and clean up old ones
            state.setdefault("last_state_clear_date", None)
            state.setdefault("last_trade_feedback", "NONE") 
            # Clean up obsolete time-based keys if they exist in the file
            if "signal_iterations" in state: del state["signal_iterations"]
            if "last_signal_price" in state: del state["last_signal_price"]
            logger.info(f"State loaded successfully. Last signal: {state.get('last_signal')}")
            return state
    except Exception as e:
        logger.error(f"❌ Error loading state file, resetting state: {e}")
        return {"last_signal": None, "prices": [], "last_state_clear_date": None, "last_trade_feedback": "NONE"}

def save_state(state: Dict[str, Any]):
    """Saves price history and last signal to state file."""
    try:
        # Clean obsolete keys before saving
        if "signal_iterations" in state: del state["signal_iterations"]
        if "last_signal_price" in state: del state["last_signal_price"]
        
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
        
        # New clean state 
        new_state = {"last_signal": None, "prices": [], "last_state_clear_date": today_date_str, "last_trade_feedback": "NONE"}
        save_state(new_state)
        return new_state
        
    return state
    
def calculate_max_pain_and_pcr(option_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Max Pain and Put-Call Ratio (PCR) from the full option chain data."""
    if not option_data:
        return {"max_pain": "N/A", "pcr": 0.0}

    # 1. Calculate PCR
    total_put_oi = 0
    total_call_oi = 0
    for record in option_data:
        if record.get('PE') and isinstance(record['PE'].get('openInterest'), (int, float)):
            total_put_oi += record['PE']['openInterest']
        if record.get('CE') and isinstance(record['CE'].get('openInterest'), (int, float)):
            total_call_oi += record['CE']['openInterest']
    
    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0.0

    # 2. Calculate Max Pain (Logic confirmed as mathematically correct)
    max_pain = "N/A"
    min_loss = float('inf')
    
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

        filtered_records = [
            record for record in full_option_data 
            if record['strikePrice'] in strikes_needed and record['expiryDate'] == expiry
        ]
        
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


def get_ai_trade_suggestion(option_chain_data: List[Dict[str, Any]], price: float, sma9: float, sma21: float, signal_type: str, pcr: float, max_pain: str, last_trade_feedback: str) -> str:
    """
    Evaluates a trading signal and Nifty Options Chain using the Gemini API.
    (Removed Time-Based Exit parameters from signature)
    """
    if not client:
        return "AI error: Gemini client is not initialized."

    option_chain_str = prepare_gemini_prompt(option_chain_data)
    
    expiry_date = option_chain_data[0].get('expiryDate', 'N/A') if option_chain_data else 'N/A'
    
    # --- FINAL PROMPT WITH ALL CONSTRAINTS AND FEEDBACK ---
    user_prompt = f"""
**SYSTEM PROMPT: You are a highly specialized and experienced NIFTY options market analyst and strategist. Your goal is to combine technical (SMA), PCR, Max Pain (Bias), and New Writing (Conviction) to generate a single, actionable, risk-managed trading recommendation.**

Input Data:
Signal: {signal_type}
Spot Price: {price:.2f}
SMA9: {sma9:.2f}
SMA21: {sma21:.2f}
Current UTC Date: {datetime.datetime.now(datetime.timezone.utc).date().isoformat()}
Option Expiry Date: {expiry_date}
Put-Call Ratio (PCR): {pcr:.2f}
Max Pain Level: {max_pain}
Last Trade Feedback: {last_trade_feedback}
Option Chain Data (Filtered JSON):
{option_chain_str}

--- GUIDELINES AND CONSTRAINTS ---

1.  **Definitions & Data Constraint:**
    * **Resistance/Support Targets (TP/SL):** MUST be based only on strikes with the highest **Open Interest (OI)** or **Change in OI (Chg in OI)**.
    * **Strike Price** and **NIFTY Price Levels (TP/SL)** MUST be selected ONLY from the strike prices provided in the 'Option Chain Data' JSON. DO NOT create a numerical value that is not present.
2.  **Trade Parameters:**
    * **Take Profit (TP) Target** MUST be set at the nearest strong **New Writing Resistance (CE OI)** or **Support (PE OI)** level that aligns with the signal.
    * **Stop Loss (SL) Target** MUST be set at the nearest strong **Conflicting New Writing** level.
    * **MAX PAIN MUST NOT BE USED TO SET TP OR SL. It is for bias check only.**

3.  **MARKET STRUCTURE (PCR/NEW WRITING) ANALYSIS:**
    * **New Writing (Conviction):** The AI must prioritize signals confirmed by new writing (high Chg in OI) over all other OI metrics.
    * **PCR/Bias:** Use PCR (0.7-1.3 neutral zone) and Max Pain as secondary directional confirmation only.

4.  **VOLATILITY AND EXPIRY DAY RULE:**
    * **Confidence Level** can be: **(Very High, High, Medium, or Low).**
    * If today's date matches the Option Expiry Date ({expiry_date}), automatically apply a **one-tier downgrade** to the initial Confidence Level.

5.  **FEEDBACK REINFORCEMENT:**
    * If **Last Trade Feedback** was 'NEGATIVE', automatically downgrade the initial Confidence of the current signal by one tier (reflecting recent adverse conditions/strategy failure).
    * If **Last Trade Feedback** was 'POSITIVE', automatically upgrade the initial Confidence of the current signal by one tier (reflecting recent success/favorable conditions).

6.  **R/R Constraint (R/R > 1.5):**
    * If the calculated Risk/Reward (R/R) ratio is less than 1.5, the final confidence MUST be **Low**.

--- REQUIRED OUTPUT FORMAT ---

**Output MUST be a single, continuous line of plain text.**
**Output MUST contain ALL of the following key-value pairs in the exact order shown below.**
**The Reason MUST be a single, concise sentence that justifies the decision by referencing the SMA, the NEW WRITING conviction, and the PCR/Max Pain bias.**

Example desired format:
Confidence: Medium. Signal: Buy. Strike Price: 26000. Option: CE. Take Profit (TP): 26100. Stop Loss (SL): 25950. Reason: SMA confirms BUY, but confidence downgraded due to negative feedback from last trade.
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
        is_new_signal = False

        if current_trend == "buy" and state["last_signal"] != "buy":
            is_new_signal = True
            signal = "BUY"
            state["last_signal"] = "buy"
            # Time-based keys removed: state["signal_iterations"] = 0 
            # state["last_signal_price"] = price

        elif current_trend == "sell" and state["last_signal"] != "sell":
            is_new_signal = True
            signal = "SELL"
            state["last_signal"] = "sell"
            # Time-based keys removed: state["signal_iterations"] = 0 
            # state["last_signal_price"] = price
        
        # 4. If Signal Generated or Active Trade, Run AI Analysis
        if is_new_signal or state["last_signal"]: # Run AI analysis on every iteration if a signal is active
            
            # --- CRITICAL FIX: Send initial alert immediately if new signal ---
            if is_new_signal:
                 logger.critical(f"🚨 MAJOR SIGNAL DETECTED: {signal} at Price {price:.2f}")
                 send_telegram(f"*🚨 MAJOR SIGNAL DETECTED: {signal}* (Price: {price:.2f})")
            # --- END CRITICAL FIX ---


            option_chain_result = get_nifty_strikes_for_expiry()
            
            if option_chain_result and option_chain_result['records']:
                ai_result = get_ai_trade_suggestion(
                    option_chain_data=option_chain_result['records'], 
                    price=price, 
                    sma9=sma9, 
                    sma21=sma21, 
                    signal_type=state["last_signal"] or "NEUTRAL", 
                    pcr=option_chain_result['pcr'],
                    max_pain=str(option_chain_result['max_pain']),
                    last_trade_feedback=state.get("last_trade_feedback", "NONE") # PASS FEEDBACK
                )
                
                # Only send a telegram alert on a NEW signal OR if the AI suggests an EXIT (Low Confidence)
                ai_dict = {}
                try:
                    # Attempt to parse the AI output to check Confidence
                    parts = ai_result.split('. ')
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            ai_dict[key.strip()] = value.strip().replace('.', '')
                except:
                    pass # Ignore parsing errors

                log_message = ai_result.strip().replace('\n', ' | ')
                logger.critical(f"🤖 AI RECOMMENDS: {log_message}")

                current_confidence = ai_dict.get('Confidence')

                # Logic for Secondary Alerts and State Reset
                if current_confidence == 'Low' or current_confidence == 'LOWEST':
                    
                    # 1. New Signal REJECTED (R/R or Structural violation)
                    if is_new_signal:
                         send_telegram(f"*⚠️ Signal Rejected:* {ai_result}")
                         
                         # CRITICAL FIX: Preserve last_signal state to prevent spamming until SMA flips.
                         # Only record the failure.
                         state["last_trade_feedback"] = "NEGATIVE" # Record rejection as negative feedback
                    
                    # 2. Active Trade (Only alert once if confidence drops)
                    elif state["last_signal"]: 
                         # This alerts the user that the active trade is no longer viable (R/R violation)
                         send_telegram("*🛑 VIOLATION EXIT:* " + ai_result)
                         
                         # Reset state after forced exit due to R/R violation
                         state["last_signal"] = None
                         state["last_trade_feedback"] = "NEGATIVE" 

                # If a signal is active and confidence is MEDIUM or HIGH, send the AI Analysis update
                elif state["last_signal"] and not is_new_signal:
                    send_telegram("*🤖 AI Analysis:* " + ai_result)
            
            else:
                logger.warning(f"⚠️ Failed to fetch valid Options Chain. Skipping AI analysis.")

        # 5. Save State
        save_state(state)

    except Exception as e:
        logger.exception(f"🔥 UNHANDLED ERROR IN MAIN LOOP: {e}")
        send_telegram(f"*🔥 FATAL ERROR in Trading Bot:*\n`{e}`")

    # Sleep until the next iteration
    time.sleep(SLEEP_SECONDS)