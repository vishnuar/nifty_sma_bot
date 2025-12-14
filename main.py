import requests
import time
import json
import datetime
import os
import logging
import sys
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type 
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------
## ⚙️ LOGGING SETUP (Console Only)
# ---------------------------------------------------------
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


#### - UPSTOX CONFIG - ####

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
NIFTY_INSTRUMENT_KEY = 'NSE_INDEX|Nifty 50'
ATM_STRIKES_TO_FETCH = int(os.getenv("STRIKES_TO_FETCH"))
CONTRACT_API_URL = 'https://api.upstox.com/v2/option/contract'
OPTION_CHAIN_API_URL = 'https://api.upstox.com/v2/option/chain'

def get_api_headers(access_token: str) -> dict:
    """Returns standard headers for Upstox API calls."""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

#### - UPSTOX CONFIG - ####

# Gemini API is used for AI Analysis
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash" 

STATE_FILE = "state.json"
MARKET_CLOSE_HOUR_UTC = 10 # 3:30 PM IST is 10:00 AM UTC
MARKET_CLOSE_MINUTE_UTC = 0

# --- NEW CONFIGURATION: SMA Buffer (Read from environment) ---
try:
    SMA_BUFFER_POINTS = float(os.getenv("SMA_BUFFER"))
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
    market_open_utc = datetime.time(3, 30) 
    # 3:30 PM IST is 10:00 UTC
    market_close_utc = datetime.time(MARKET_CLOSE_HOUR_UTC, MARKET_CLOSE_MINUTE_UTC)

    return market_open_utc <= now_utc.time() <= market_close_utc

def fetch_closest_expiry(access_token: str) -> str | None:
    """
    Step 1: Calls the /option/contract API to find the nearest Nifty expiry date.
    """
    print("1. Fetching all Nifty option contracts to find the closest expiry...")
    
    headers = get_api_headers(access_token)
    params = {'instrument_key': NIFTY_INSTRUMENT_KEY}

    try:
        response = requests.get(CONTRACT_API_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'success' and data.get('data'):
            today = datetime.datetime.now().date()
            expiry_dates = set()
            for contract in data['data']:
                expiry_str = contract.get('expiry')
                if expiry_str:
                    try:
                        expiry_date = datetime.datetime.strptime(expiry_str, '%Y-%m-%d').date()
                        if expiry_date >= today:
                            expiry_dates.add(expiry_str)
                    except ValueError:
                        continue
            
            if expiry_dates:
                closest_expiry = min(expiry_dates, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
                print(f"   -> Closest Expiry Date found: {closest_expiry}")
                return closest_expiry
            else:
                print("   -> No future expiry dates found.")
                return None
        else:
            print(f"   -> Error or empty response from contract API: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   -> API Request Error in step 1: {e}")
        return None

def normalize_upstox_records(raw_upstox_records: list, expiry_date: str) -> List[Dict[str, Any]]:
    """
    Transforms the verbose Upstox API records into a simplified list 
    containing only the fields required by the Gemini AI prompt.
    """
    ai_prompt_records = []
    
    for item in raw_upstox_records:
        ce_data = item['call_options']
        pe_data = item['put_options']
        
        # --- CALCULATE CHANGE IN OI MANUALLY ---
        ce_oi_current = ce_data['market_data'].get('oi', 0)
        ce_oi_prev = ce_data['market_data'].get('prev_oi', 0)
        
        pe_oi_current = pe_data['market_data'].get('oi', 0)
        pe_oi_prev = pe_data['market_data'].get('prev_oi', 0)
        
        # Delta OI = Current OI - Previous OI
        ce_change_in_oi = ce_oi_current - ce_oi_prev
        pe_change_in_oi = pe_oi_current - pe_oi_prev
        # --- END MANUAL CALCULATION ---
        
        # CRITICAL: This is the structure required by the AI prompt
        record = {
            "strikePrice": item.get('strike_price', 0.0),
            "expiryDate": expiry_date,
            "CE": {
                "openInterest": ce_oi_current,
                "changeinOpenInterest": ce_change_in_oi,
                "Delta": ce_data['option_greeks'].get('delta', 0.0),
                "IV": ce_data['option_greeks'].get('iv', 0.0)                
            },
            "PE": {
                "openInterest": pe_oi_current,
                "changeinOpenInterest": pe_change_in_oi,
                "Delta": pe_data['option_greeks'].get('delta', 0.0),
                "IV": pe_data['option_greeks'].get('iv', 0.0)                
            }
        }
        ai_prompt_records.append(record)
        
    return ai_prompt_records


def fetch_and_filter_option_chain(expiry_date: str, access_token: str, num_strikes: int):
    """
    Step 2: Fetches the full option chain, filters for ATM contracts.
    Returns data in the user-requested format.
    """
    print(f"2. Fetching full Option Chain for Expiry: {expiry_date}...")

    headers = get_api_headers(access_token)
    params = {
        'instrument_key': NIFTY_INSTRUMENT_KEY,
        'expiry_date': expiry_date
    }

    try:
        response = requests.get(OPTION_CHAIN_API_URL, params=params, headers=headers)
        response.raise_for_status()
        chain_data = response.json()

        if chain_data.get('status') != 'success' or not chain_data.get('data'):
            print(f"   -> Error fetching Option Chain: {chain_data.get('message', 'No data returned')}")
            return {'status': 'error', 'message': chain_data.get('message', 'No data returned')}

        all_strikes = [item['strike_price'] for item in chain_data['data']]
        all_strikes.sort()
        
        spot_price = chain_data['data'][0].get('underlying_spot_price', 'N/A')
        
        if not all_strikes:
            return {'status': 'error', 'message': 'No strike prices available.'}
            
        atm_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
        atm_index = all_strikes.index(atm_strike)

        start_index = max(0, atm_index - num_strikes)
        end_index = min(len(all_strikes), atm_index + num_strikes + 1)
        
        selected_strikes_list = sorted(all_strikes[start_index:end_index])
        selected_strikes_set = set(selected_strikes_list)
        
        filtered_chain = [
            item for item in chain_data['data'] 
            if item['strike_price'] in selected_strikes_set
        ]
        
        ai_prompt_records = normalize_upstox_records(filtered_chain, expiry_date)
        
        return {
            "spot": spot_price,
            "atm": atm_strike,
            "expiry": expiry_date,       
            "records": ai_prompt_records
        }

    except requests.exceptions.RequestException as e:
        print(f"   -> API Request Error in step 2: {e}")
        return {'status': 'error', 'message': f'API Request Error: {e}'}

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
    logger.debug("Executing Gemini API call...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    logger.debug("Gemini call completed successfully.")
    return response.text


def get_ai_trade_suggestion(option_chain_data: List[Dict[str, Any]], price: float, sma9: float, sma21: float, signal_type: str, expiry_date: str) -> str:
    if not client:
        return "AI error: Gemini client is not initialized."

    option_chain_str = prepare_gemini_prompt(option_chain_data)

    user_prompt =  f"""
**SYSTEM PROMPT: You are a highly specialized and experienced NIFTY options market analyst and strategist. Your role is to treat the SMA signal ONLY as a trigger alert. Your primary function is to analyze the Option Chain data (OI, Delta, IV, and structural checks) to determine trade quality and conviction, generating a single, actionable, risk-managed trading recommendation.**

Input Data:
Signal: {signal_type}
Spot Price: {price:.2f}
SMA9: {sma9:.2f}
SMA21: {sma21:.2f}
Current UTC Date: {datetime.datetime.now(datetime.timezone.utc).date().isoformat()}
Option Chain Data (Filtered JSON):
{option_chain_str}

--- 📋 GUIDELINES AND CONSTRAINTS 📋 ---

1. **Definitions & Data Constraint 🧱:**
* 📉 **Resistance (TP Target):** Strong Call Option (CE) Open Interest (OI) or Change in OI build-up.
* 🛡️ **Support (SL Target for BUY/TP Target for SELL):** Strong Put Option (PE) Open Interest (OI) or Change in OI build-up.
* 🎯 **Strike Price** and **NIFTY Price Levels (TP/SL)** MUST be selected ONLY from the strike prices provided in the 'Option Chain Data' JSON. ❌ DO NOT create a numerical value that is not present.

2. **Critical Options Metrics (Delta/IV) 🧪:**
* ☢️ **IV Check (Risk Filter):** The selected strike's Implied Volatility (IV) MUST be evaluated. If the IV for the suggested option (CE for BUY, PE for SELL) is **above 150.0**, the AI MUST **apply a one-tier downgrade** ⬇️ to the assigned Confidence Level.
* 🔺 **Delta Selection:** When multiple strikes offer a similar OI advantage, prioritize the strike whose Delta is closest to 0.50 for higher responsiveness and probability.

3. **Trade Parameters (Dominance & Structural Checks) ⚖️:**
* ⬆️ **Take Profit (TP) Target):** MUST be the strike with the **highest NET OI and Chg in OI** in the favorable direction.
* ⬇️ **Stop Loss (SL) Target):** MUST be the strike with the **highest NET OI and Chg in OI** in the opposite direction.

4. **Market Structure Analysis (Primary Focus) 🔬:**
* ✍️ **New Writing (Conviction):** The AI must prioritize signals confirmed by new writing over other OI metrics.
* ⚓ **NEW WRITING CONFIRMATION (CRITICAL - AT LEAST ONE ANCHOR):**
  * 🟢 **For BUY Signal (OR Logic):** The trade requires at least one strong anchor. This condition is **MET** if **EITHER** the **New Writing (Change in OI) on the CE side** is **actively increasing and positive** (confirming bearish defense), **OR** the **Put Option (PE) Open Interest (OI) at the entry strike** is judged to be a **strong support level** (i.e., above the average OI of the nearest 4 strikes). If **BOTH** conditions fail, the AI MUST apply a **one-tier downgrade** ⬇️.
  * 🔴 **For SELL Signal (OR Logic):** The trade requires at least one strong anchor. This condition is **MET** if **EITHER** the **New Writing (Change in OI) on the PE side** is **actively increasing and positive** (confirming bullish support), **OR** the **Call Option (CE) Open Interest (OI) at the entry strike** is judged to be a **strong resistance level** (i.e., above the average OI of the nearest 4 strikes). If **BOTH** conditions fail, the AI MUST apply a **one-tier downgrade** ⬇️.

5. **Confidence (Points-Based System) ⭐:**
* 💰 **Reward Calculation:** The Reward is the absolute distance between the Take Profit (TP) Strike Price and the Entry/Strike Price.
* 🏆 **Confidence Level** can be: **(Very High, High, Medium, or Low).**
* ✨ **Initial Assignment (Based on Reward Points):**
 1. If Reward is **101 points or more**, start with **Very High** ⏫ Confidence.
 2. If Reward is between **50 and 100 points**, start with **High** ⬆️ Confidence.
 3. If Reward is between **25 and 49 points**, start with **Medium** ↔️ Confidence.
 4. If Reward is **less than 25 points**, the final confidence MUST be **Low** ⬇️ (Ultimate Risk Filter).
* 📉 **Risk Filters (Downgrade Rules):** The Confidence level assigned above MUST be downgraded based on any rule violations in sections 2, 4, and 5. **(The Confidence is primarily determined by the structural quality of the option chain, not the SMA alert.)**

--- REQUIRED OUTPUT FORMAT ---

**Output MUST be a single, continuous line of plain text and layman words**
**Output MUST contain ALL of the following key-value pairs in the exact order shown below.**
**The Reason MUST be a single, concise sentence that justifies the decision by referencing the Option Chain structure (New Writing, OI levels, Delta), and must only mention the SMA as a contextual factor.**

Example desired format:
Confidence: ⭐High. Signal: 🟢Buy. Strike Price: 🎯{{Calculated_Strike_Price}}. Option: CE. Take Profit (TP): ⬆️{{Calculated_TP_Level}}. Stop Loss (SL): ⬇️{{Calculated_SL_Level}}. Max Resistance: 🛑{{Calculated_Max_Resistance}}. Max Support: ✅{{Calculated_Max_Support}}. Reason: Strong new CE writing confirms bearish defense (primary conviction), TP is set by major OI at {{Calculated_TP_Level}}, Delta {{Calculated_Delta}} indicates strong probability, and SMA provides the initial alert.
"""
    try:
        logger.info("Starting Gemini API call (up to 5 attempts with backoff)...")
        response_text = _call_gemini_with_retry(
            client=client,
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": "You are a highly specialized and experienced NIFTY options market analyst and strategist. Your sole function is to combine Open Interest (OI) data, Delta, and IV to generate a single, actionable, risk-managed trading recommendation"}]},
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
        SMA_BUFFER = SMA_BUFFER_POINTS

        if sma9 > (sma21 + SMA_BUFFER):
            current_trend = "buy"
        elif sma9 < (sma21 - SMA_BUFFER):
            current_trend = "sell"
        else:
            logger.info("Neutral trend detected. Keep contuning the same trend.")
            time.sleep(PRICE_FETCH_DELAY)
            continue

        if current_trend == "buy" and state["last_signal"] != "buy":
            signal = "BUY"
            state["last_signal"] = "buy"

        elif current_trend == "sell" and state["last_signal"] != "sell":
            signal = "SELL"
            state["last_signal"] = "sell"

        if signal:
            logger.critical(f"🚨 MAJOR SIGNAL DETECTED: {signal} at Price {price:.2f}")
            send_telegram(f"*🚨 Major Signal Detected: {signal}* (Price: {price:.2f})")

            closest_expiry = fetch_closest_expiry(UPSTOX_ACCESS_TOKEN)

            if not closest_expiry:
                logger.warning("⚠️ Could not determine closest expiry — skipping option-chain fetch and AI analysis.")
                send_telegram(f"*⚠️ Warning:* Could not determine closest expiry — skipping option-chain fetch and AI analysis.")
            else:
                option_chain_result = fetch_and_filter_option_chain(
                    expiry_date=closest_expiry,
                    access_token=UPSTOX_ACCESS_TOKEN,
                    num_strikes=ATM_STRIKES_TO_FETCH)

                if option_chain_result and option_chain_result.get('records'):
                    ai_result = get_ai_trade_suggestion(
                        option_chain_data=option_chain_result['records'], 
                        price=price, 
                        sma9=sma9, 
                        sma21=sma21, 
                        signal_type=signal,
                        expiry_date=closest_expiry
                    )
                    ai_log_message = ai_result.strip().replace('\n', ' | ')
                    logger.critical(f"🤖 AI RECOMMENDS: {ai_log_message}")
                    send_telegram("*🤖 AI Analysis:*\n" + ai_result)
                else:
                    logger.warning(f"⚠️ Failed to fetch valid Options Chain for {signal} signal. Skipping AI analysis.")
                    send_telegram(f"*⚠️ Warning:* Failed to fetch valid Options Chain data for {signal} signal.")

        save_state(state)
        time.sleep(PRICE_FETCH_DELAY)


    except Exception as e:
        logger.exception(f"🔥 UNHANDLED ERROR IN MAIN LOOP: {e}")
        send_telegram(f"*🔥 FATAL ERROR in Trading Bot:*\n`{e}`")
