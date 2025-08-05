from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
import re
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import logging
from datetime import datetime, timedelta
import json
from logging.handlers import WatchedFileHandler
import sys

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# File handler
handler = WatchedFileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Suppress PIL sBIT warnings
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
# Suppress Matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.INFO)

# Check for openpyxl (for environment clarity, though unused)
try:
    import openpyxl
    logger.debug("openpyxl is installed and imported successfully")
except ImportError:
    logger.warning("openpyxl is not installed, but not required for this app")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

# Configuration
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
NEWS_API_KEY = "9d2d1619e522462fb62b0366f23b7841"
NEWS_CACHE_FILE = os.path.join(PROJECT_DIR, "news_cache.json")
NEWS_CACHE_DURATION = 3600  # Cache news for 1 hour
PORTFOLIO_FILE = os.path.join(PROJECT_DIR, "portfolio.json")
ALERTS_FILE = os.path.join(PROJECT_DIR, "alerts.json")
PREDICTIONS_LOG_FILE = os.path.join(PROJECT_DIR, "predictions_log.json")

cryptos = ["bitcoin", "ethereum", "binancecoin", "cardano"]
crypto_aliases = {
    "bitcoin": ["bitcoin", "btc"],
    "ethereum": ["ethereum", "eth"],
    "binancecoin": ["binancecoin", "bnb"],
    "cardano": ["cardano", "ada"]
}
models = {}
historical_data = {}
scalers = {}
y_scalers = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/news")
def news():
    crypto = request.args.get("crypto", None)
    page = int(request.args.get("page", 1))
    articles = fetch_news(crypto, page=page)
    return render_template("news.html", articles=articles, cryptos=cryptos, selected_crypto=crypto, page=page)

@app.route("/portfolio")
def portfolio():
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            holdings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        holdings = []
    total_value = 0
    portfolio_data = []
    for holding in holdings:
        price, change = fetch_live_price(holding["crypto"])
        if price:
            value = price * holding["amount"]
            total_value += value
            profit_loss = (price - holding["purchase_price"]) * holding["amount"]
            portfolio_data.append({
                "crypto": holding["crypto"].capitalize(),
                "amount": holding["amount"],
                "purchase_price": holding["purchase_price"],
                "current_price": round(float(price), 2),
                "value": round(float(value), 2),
                "profit_loss": round(float(profit_loss), 2)
            })
    return render_template("portfolio.html", portfolio=portfolio_data, total_value=round(float(total_value), 2))

@app.route("/alerts")
def alerts():
    try:
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        alerts = []
    return render_template("alerts.html", alerts=alerts)

@app.route("/debug")
def debug():
    return jsonify({
        "historical_data_keys": list(historical_data.keys()),
        "model_keys": list(models.keys()),
        "y_scaler_keys": list(y_scalers.keys()),
        "scaler_keys": list(scalers.keys())
    })

@app.route("/predict", methods=["GET"])
def predict():
    crypto = request.args.get("crypto")
    try:
        days = int(request.args.get("days", 1))
    except ValueError:
        logger.error(f"Invalid days parameter: {request.args.get('days')}")
        return jsonify({"error": "Days must be a valid integer"}), 400

    if crypto not in cryptos:
        logger.error(f"Unsupported cryptocurrency: {crypto}")
        return jsonify({"error": f"Invalid cryptocurrency: {crypto}"}), 400

    if crypto not in models or crypto not in historical_data or crypto not in scalers or crypto not in y_scalers:
        logger.error(f"Missing data/model/scaler for cryptocurrency: {crypto}")
        return jsonify({"error": f"Data or model not initialized for {crypto}. Try again later."}), 500

    if days < 1 or days > 30:
        logger.error(f"Days out of range: {days}")
        return jsonify({"error": "Days must be between 1 and 30"}), 400

    try:
        df = historical_data[crypto]
        if df.empty or "day" not in df.columns or "price" not in df.columns or "date" not in df.columns:
            logger.error(f"Invalid historical data for {crypto}")
            return jsonify({"error": f"Invalid historical data for {crypto}"}), 500

        # Validate date column
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            logger.warning(f"Date column for {crypto} is not datetime. Converting.")
            df["date"] = pd.to_datetime(df["date"])

        last_day = df["day"].max()
        future_day = last_day + days

        try:
            last_date = pd.to_datetime(df["date"].max())
            prediction_date = last_date + timedelta(days=days)
            actual_date = datetime.now()
        except Exception as e:
            logger.warning(f"Failed to calculate prediction date for {crypto}: {str(e)}. Falling back to current date.")
            prediction_date = datetime.now() + timedelta(days=days)
            actual_date = datetime.now()
        logger.debug(f"Prediction date for {crypto}: {prediction_date.strftime('%Y-%m-%d')} (days ahead: {days})")

        last_row = df.iloc[-1]
        last_price = float(last_row["price"])
        recent_prices = df["price"].tail(7) if len(df) >= 7 else df["price"]
        daily_change = float((recent_prices.iloc[-1] - recent_prices.iloc[0]) / max(len(recent_prices) - 1, 1)) if len(recent_prices) > 1 else 0
        lag1 = float(last_price + daily_change * (days - 1))
        lag7 = float(df.iloc[-7]["price"]) if len(df) >= 7 else last_price
        ma7 = float(recent_prices.mean() + daily_change * days / 2)

        logger.debug(f"Future features for {crypto} (days={days}): day={future_day}, lag1={lag1}, lag7={lag7}, ma7={ma7}")
        future_features = np.array([[future_day, lag1, lag7, ma7]])
        scaled_future_features = scalers[crypto].transform(future_features)
        logger.debug(f"Scaled future features for {crypto}: {scaled_future_features}")

        prediction_scaled = models[crypto].predict(scaled_future_features)
        predicted_price = float(y_scalers[crypto].inverse_transform(prediction_scaled.reshape(-1, 1))[0][0])

        actual_price, _ = fetch_live_price(crypto)
        if actual_price is None:
            logger.warning(f"Failed to fetch live price for {crypto}, falling back to historical price")
            actual_price = last_price
        logger.debug(f"Predicted price for {crypto} in {days} days (on {prediction_date.strftime('%Y-%m-%d')}): {predicted_price}, Actual price: {actual_price}")

        if days > 1:
            base_features = np.array([[last_day + 1, last_price, lag7, float(recent_prices.mean())]])
            base_scaled = scalers[crypto].transform(base_features)
            base_pred = float(y_scalers[crypto].inverse_transform(models[crypto].predict(base_scaled).reshape(-1, 1))[0][0])
            if abs(predicted_price - base_pred) < 1.0:
                logger.warning(f"Prediction for {crypto} in {days} days is too similar to days=1: {predicted_price} vs {base_pred}")

        accuracy = 0 if actual_price == 0 else float(100 - (abs((actual_price - predicted_price) / actual_price) * 100))

        log_prediction(crypto, days, predicted_price, actual_price, accuracy, model_day_index=future_day, prediction_date=prediction_date)

        historical_prices = [
            {"day": int(row["day"]), "price": float(row["price"]), "date": row["date"].strftime("%Y-%m-%d")}
            for row in df[["day", "price", "date"]].tail(30).to_dict(orient="records")
        ]

        # Log historical_prices for debugging
        logger.debug(f"Historical prices for {crypto}: {json.dumps(historical_prices, indent=2)}")

        # Validate historical_prices
        if not historical_prices:
            logger.error(f"No historical prices available for {crypto}")
            return jsonify({"error": f"No historical price data for {crypto}"}), 500
        for entry in historical_prices:
            if not isinstance(entry["date"], str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", entry["date"]):
                logger.error(f"Invalid date format in historical_prices for {crypto}: {entry['date']}")
                return jsonify({"error": f"Invalid date format in historical data for {crypto}"}), 500

        try:
            with open(ALERTS_FILE, "r") as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []
        triggered = []
        for alert in alerts:
            if alert["crypto"] == crypto:
                target = alert["target_price"]
                if (alert["direction"] == "above" and actual_price >= target) or (alert["direction"] == "below" and actual_price <= target):
                    triggered.append(f"Alert: {crypto.capitalize()} has {alert['direction']} ${target}! Current price: ${round(float(actual_price), 2)}")

        response = {
            "predicted_price": round(float(predicted_price), 2),
            "actual_price": round(float(actual_price), 2),
            "actual_date": actual_date.strftime("%Y-%m-%d") + " (Today)",
            "predicted_date": prediction_date.strftime("%Y-%m-%d"),
            "accuracy_percentage": round(float(accuracy), 2),
            "historical_prices": historical_prices,
            "future_day": int(future_day)
        }
        if triggered:
            response["alerts"] = triggered
        logger.debug(f"Prediction response for {crypto}: {json.dumps(response, indent=2)}")
        return jsonify(response)
    except PermissionError as e:
        logger.error(f"Permission error in /predict for {crypto} in {days} days: {str(e)}")
        return jsonify({"error": f"Permission error saving prediction: {str(e)}"}), 500
    except ValueError as e:
        logger.error(f"Value error in /predict for {crypto} in {days} days: {str(e)}")
        return jsonify({"error": f"Invalid data or calculation error: {str(e)}"}), 500
    except KeyError as e:
        logger.error(f"Key error in /predict for {crypto} in {days} days: {str(e)}")
        return jsonify({"error": f"Data access error: {str(e)}"}), 500
    except ZeroDivisionError as e:
        logger.error(f"Zero division error in /predict for {crypto} in {days} days: {str(e)}")
        return jsonify({"error": f"Calculation error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Prediction error in /predict for {crypto} in {days} days: {str(e)}")
        return jsonify({"error": f"Failed to generate prediction: {str(e)}"}), 500
def log_prediction(crypto, days, predicted_price, actual_price, accuracy, model_day_index=None, prediction_date=None):
    try:
        try:
            with open(PREDICTIONS_LOG_FILE, "r") as f:
                log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log = []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_day_index = model_day_index if model_day_index is not None else (historical_data[crypto]["day"].max() + days)
        # Ensure prediction_date is a datetime object
        if isinstance(prediction_date, str):
            logger.warning(f"prediction_date is a string: {prediction_date}. Converting to datetime.")
            prediction_date = pd.to_datetime(prediction_date)
        prediction_date_str = prediction_date.strftime("%Y-%m-%d") if prediction_date else (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        log_entry = {
            "timestamp": timestamp,
            "crypto": crypto,
            "days_ahead": days,
            "prediction_date": prediction_date_str,
            "model_day_index": int(model_day_index),
            "predicted_price": round(float(predicted_price), 2),
            "actual_price": round(float(actual_price), 2),
            "accuracy": round(float(accuracy), 2)
        }
        log.append(log_entry)

        log = log[-100:]

        with open(PREDICTIONS_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
        logger.debug(f"Logged prediction for {crypto} ({days} days ahead, date: {prediction_date_str}, model day: {model_day_index})")
    except Exception as e:
        logger.error(f"Error logging prediction for {crypto}: {str(e)}")
        raise

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "").strip()
        logger.debug(f"Received chat message: '{user_input}'")
        if not user_input:
            logger.warning("Empty message received")
            return jsonify({"response": "Please type a message!"}), 400
        response = get_bot_response(user_input)
        logger.debug(f"Sending response: '{response}'")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"response": "Sorry, something went wrong on the server."}), 500

def fetch_data(crypto, retries=3, delay=2):
    cache_file = os.path.join(PROJECT_DIR, f"{crypto}_data.csv")
    logger.debug(f"Attempting to load data for {crypto} from {cache_file}")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            if not df.empty and all(col in df.columns for col in ["day", "price", "date"]):
                # Verify date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                    logger.warning(f"Date column in {cache_file} is not datetime. Converting.")
                    df["date"] = pd.to_datetime(df["date"])
                logger.debug(f"Loaded cached data for {crypto} with {len(df)} rows")
                return df
            else:
                logger.warning(f"Cache file {cache_file} is invalid or missing required columns")
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {str(e)}")

    logger.debug(f"Fetching fresh data for {crypto} from CoinGecko")
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365&interval=daily"
    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, timeout=10)
            logger.debug(f"CoinGecko response status for {crypto}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                if not prices:
                    logger.error(f"No price data returned for {crypto}")
                    return None
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["day"] = np.arange(len(df))
                try:
                    df.to_csv(cache_file, index=False)
                    logger.debug(f"Saved data to {cache_file}")
                except Exception as e:
                    logger.error(f"Error saving cache file {cache_file}: {str(e)}")
                return df
            elif response.status_code == 429:
                logger.warning(f"CoinGecko rate limit exceeded for {crypto}, attempt {attempt+1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                logger.error(f"Exhausted retries for {crypto} due to rate limit")
                return None
            else:
                logger.error(f"CoinGecko API error for {crypto}: Status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {crypto}, attempt {attempt+1}/{retries}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.error(f"Exhausted retries for {crypto}")
                return None
    return None

def train_model(df, crypto):
    try:
        if df.empty or "day" not in df.columns or "price" not in df.columns:
            logger.error(f"Invalid DataFrame for {crypto}: {df.columns}")
            return None

        df = df.copy()
        df['lag1'] = df['price'].shift(1)
        df['lag7'] = df['price'].shift(7)
        df['ma7'] = df['price'].rolling(window=7).mean()
        df = df.dropna()
        logger.debug(f"Processed DataFrame for {crypto} with {len(df)} rows after feature engineering")

        if df.empty:
            logger.error(f"DataFrame for {crypto} is empty after feature engineering")
            return None

        X = df[["day", "lag1", "lag7", "ma7"]].values
        y = df[["price"]].values

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred.reshape(-1, 1))))
        logger.debug(f"Trained model for {crypto} with {len(X_train)} samples, MAE on test set: {mae:.2f}")

        scalers[crypto] = x_scaler
        y_scalers[crypto] = y_scaler
        return model
    except Exception as e:
        logger.error(f"Error training model for {crypto}: {str(e)}")
        return None

def fetch_live_price(crypto):
    try:
        time.sleep(1)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = float(data[crypto]["usd"])
            change = float(data[crypto]["usd_24h_change"])
            return price, change
        logger.error(f"Failed to fetch live price for {crypto}: Status {response.status_code}")
        return None, None
    except Exception as e:
        logger.error(f"Error fetching live price for {crypto}: {str(e)}")
        return None, None

def fetch_news(crypto=None, sources=None, from_date=None, language='en', page=1):
    try:
        cache_key = f"{crypto or 'cryptocurrency'}_{sources or 'all'}_{language}_{page}"
        if os.path.exists(NEWS_CACHE_FILE):
            with open(NEWS_CACHE_FILE, "r") as f:
                cache = json.load(f)
                if cache.get("cache_key") == cache_key and (datetime.now().timestamp() - cache["timestamp"]) < NEWS_CACHE_DURATION:
                    logger.debug(f"Returning cached news for {cache_key}")
                    return cache["articles"]

        query = crypto or "cryptocurrency"
        url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&page={page}&pageSize=10&apiKey={NEWS_API_KEY}"
        if sources:
            url += f"&sources={sources}"
        if from_date:
            url += f"&from={from_date}"
        
        logger.debug(f"NewsAPI request: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if not articles:
                logger.warning(f"No articles found for query: {query}, page: {page}")
                return []

            for article in articles:
                text = (article.get("title", "") + " " + article.get("description", "")).lower()
                positive = len(re.findall(r"\b(good|rise|bullish|success|grow)\b", text))
                negative = len(re.findall(r"\b(bad|fall|bearish|crash|decline)\b", text))
                article["sentiment"] = "positive" if positive > negative else "negative" if negative > positive else "neutral"
                article["publish_date"] = article.get("publishedAt", "")[:10]

            cache_data = {
                "timestamp": datetime.now().timestamp(),
                "cache_key": cache_key,
                "articles": articles
            }
            with open(NEWS_CACHE_FILE, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Fetched and cached {len(articles)} news articles for {cache_key}")
            return articles
        elif response.status_code == 401:
            logger.error("NewsAPI unauthorized: Invalid API key")
            return [{"error": "Invalid NewsAPI key. Please check your configuration."}]
        elif response.status_code == 429:
            logger.error("NewsAPI rate limit exceeded")
            return [{"error": "NewsAPI rate limit reached. Try again later."}]
        else:
            logger.error(f"NewsAPI error: Status {response.status_code}")
            return [{"error": f"Failed to fetch news (Status {response.status_code})"}]
    except Exception as e:
        logger.error(f"News fetch error: {str(e)}")
        return [{"error": f"Failed to fetch news: {str(e)}"}]

def get_price_history(crypto, days_ago):
    try:
        if crypto not in historical_data:
            return None
        df = historical_data[crypto]
        target_date = datetime.now() - timedelta(days=days_ago)
        df["date_diff"] = abs((df["date"] - target_date).dt.total_seconds())
        closest_row = df.loc[df["date_diff"].idxmin()]
        return float(closest_row["price"])
    except Exception:
        return None

def predict_chat_response(crypto, days):
    try:
        if crypto not in historical_data or crypto not in models:
            logger.error(f"No data or model for {crypto}")
            return f"Sorry, I don't have data for {format_crypto_name(crypto)}."
        if crypto not in y_scalers or crypto not in scalers:
            logger.error(f"No scaler for {crypto}")
            return f"Sorry, scaling data is missing for {format_crypto_name(crypto)}."
        df = historical_data[crypto]
        last_day = df["day"].max()
        future_day = last_day + days

        last_row = df.iloc[-1]
        last_price = float(last_row["price"])
        recent_prices = df["price"].tail(7) if len(df) >= 7 else df["price"]
        daily_change = float((recent_prices.iloc[-1] - recent_prices.iloc[0]) / max(len(recent_prices) - 1, 1)) if len(recent_prices) > 1 else 0
        lag1 = float(last_price + daily_change * (days - 1))
        lag7 = float(df.iloc[-7]["price"]) if len(df) >= 7 else last_price
        ma7 = float(recent_prices.mean() + daily_change * days / 2)

        logger.debug(f"Future features for {crypto} (days={days}): day={future_day}, lag1={lag1}, lag7={lag7}, ma7={ma7}")
        future_features = np.array([[future_day, lag1, lag7, ma7]])
        scaled_future_features = scalers[crypto].transform(future_features)
        logger.debug(f"Scaled future features for {crypto}: {scaled_future_features}")

        prediction_scaled = models[crypto].predict(scaled_future_features)
        predicted_price = float(y_scalers[crypto].inverse_transform(prediction_scaled.reshape(-1, 1))[0][0])
        actual_price = last_price
        logger.debug(f"Predicted price: {predicted_price}, Actual price: {actual_price}")

        accuracy = 0 if actual_price == 0 else float(100 - (abs((actual_price - predicted_price) / actual_price) * 100))
        return (
            f"Based on my XGBoost model, I predict the price of {format_crypto_name(crypto)} in {days} day(s) will be around ${round(float(predicted_price), 2)} "
            f"with an estimated accuracy of {round(float(accuracy), 2)}%. Keep in mind, crypto markets are volatile, so use this as a guide, not a guarantee!"
        )
    except Exception as e:
        logger.error(f"Prediction error for {crypto} in {days} days: {str(e)}")
        return f"Sorry, I couldn't predict the price for {format_crypto_name(crypto)}. The data may be unavailable. Try checking the current price or news instead."

def get_bot_response(message):
    def format_crypto_name(crypto):
        return crypto.capitalize()

    tokens = nltk.word_tokenize(message.lower())
    logger.debug(f"Processed tokens: {tokens}")

    prediction_keywords = ["predict", "forecast", "how much", "value", "price", "future"]
    prediction_phrases = ["what will", "how will"]
    price_keywords = ["current", "live", "now", "today", "price", "value"]
    history_keywords = ["history", "past", "last", "ago", "previous"]
    news_keywords = ["news", "update", "what's new", "latest", "trends", "headlines"]
    help_keywords = ["help", "what can you do", "how to use"]
    greeting_keywords = ["hi", "hello", "hey", "yo"]
    explain_keywords = ["how", "explain", "why"]
    market_keywords = ["market", "trend", "overview"]
    portfolio_keywords = ["portfolio", "holding", "add holding", "remove holding"]
    alert_keywords = ["alert", "notify", "watch"]

    crypto = None
    for token in tokens:
        for crypto_name, aliases in crypto_aliases.items():
            if token in aliases:
                crypto = crypto_name
                break
        if crypto:
            break

    days = None
    days_match = re.search(r'\b(\d+)\s*(day|days)\b', message.lower())
    if days_match:
        days = int(days_match.group(1))

    message_lower = message.lower()
    has_prediction_phrase = any(phrase in message_lower for phrase in prediction_phrases)
    has_prediction_keyword = any(keyword in tokens for keyword in prediction_keywords)

    source_match = re.search(r"\bfrom\s+(\w+(?:\s+\w+)*)\b", message_lower)
    date_match = re.search(r"\b(yesterday|last\s+\w+|today)\b", message_lower)

    intent = None
    if (has_prediction_phrase or has_prediction_keyword) and crypto and days:
        intent = "prediction"
        logger.debug(f"Intent: prediction (crypto={crypto}, days={days})")
    elif any(keyword in tokens for keyword in price_keywords) and crypto:
        intent = "price"
        logger.debug(f"Intent: price (crypto={crypto})")
    elif any(keyword in tokens for keyword in history_keywords) and crypto and days:
        intent = "history"
        logger.debug(f"Intent: history (crypto={crypto}, days={days})")
    elif any(keyword in tokens for keyword in news_keywords):
        intent = "news"
        sources = source_match.group(1).replace(" ", "-") if source_match else None
        from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") if date_match and "yesterday" in message_lower else None
        logger.debug(f"Intent: news (crypto={crypto}, sources={sources}, from_date={from_date})")
    elif any(keyword in tokens for keyword in greeting_keywords):
        intent = "greeting"
        logger.debug("Intent: greeting")
    elif any(keyword in tokens for keyword in help_keywords):
        intent = "help"
        logger.debug("Intent: help")
    elif any(keyword in tokens for keyword in explain_keywords) and any(word in tokens for word in ["predict", "forecast"]):
        intent = "explain"
        logger.debug("Intent: explain")
    elif any(keyword in tokens for keyword in market_keywords):
        intent = "market"
        logger.debug("Intent: market")
    elif any(keyword in message_lower for keyword in portfolio_keywords):
        intent = "portfolio"
        logger.debug(f"Intent: portfolio (crypto={crypto})")
    elif any(keyword in message_lower for keyword in alert_keywords):
        intent = "alert"
        logger.debug(f"Intent: alert (crypto={crypto})")

    if intent == "greeting":
        return "Hey there! I'm your Crypto Assistant, here to help with prices, predictions, news, portfolio, alerts, and more. What's on your mind?"
    elif intent == "help":
        return (
            "I'm your Crypto Assistant! You can ask me things like:\n"
            "- 'Predict BTC price in 5 days'\n"
            "- 'What's the current price of ETH?'\n"
            "- 'What was Bitcoin's price 7 days ago?'\n"
            "- 'Latest news on Cardano'\n"
            "- 'Add 0.5 BTC at $80000'\n"
            "- 'Alert me if BTC reaches above $100000'\n"
            "- 'How does the prediction work?'\n"
            "- 'Give me a market overview'\n"
            "Just type your question, and I'll get right to it!"
        )
    elif intent == "explain":
        return (
            "Our predictions are powered by XGBoost, a machine learning model trained on 365 days of historical price data from CoinGecko. "
            "It analyzes trends to forecast future prices. The accuracy depends on market conditions, but we aim to give you a solid estimate. "
            "Try asking, 'Predict Ethereum in 3 days' to see it in action!"
        )
    elif intent == "prediction":
        if crypto in cryptos and 1 <= days <= 30:
            return predict_chat_response(crypto, days)
        return f"Please specify a valid cryptocurrency ({', '.join(format_crypto_name(c) for c in cryptos)}) and days (1-30)."
    elif intent == "price":
        if crypto in cryptos:
            price, change = fetch_live_price(crypto)
            if price is not None:
                change_text = f" (24hr change: {round(float(change), 2)}%)" if change is not None else ""
                return f"The current price of {format_crypto_name(crypto)} is ${round(float(price), 2)}{change_text}. Want to predict its future price?"
            return f"Sorry, I couldn't fetch the live price for {format_crypto_name(crypto)}. Try again later."
        return f"Please specify a valid cryptocurrency ({', '.join(format_crypto_name(c) for c in cryptos)})."
    elif intent == "history":
        if crypto in cryptos and 1 <= days <= 365:
            price = get_price_history(crypto, days)
            if price is not None:
                return f"About {days} day(s) ago, the price of {format_crypto_name(crypto)} was approximately ${round(float(price), 2)}."
            return f"Sorry, I couldn't fetch the historical price for {format_crypto_name(crypto)}."
        return f"Please specify a valid cryptocurrency ({', '.join(format_crypto_name(c) for c in cryptos)}) and days (1-365)."
    elif intent == "news":
        articles = fetch_news(crypto, sources, from_date)
        if articles and not articles[0].get("error"):
            response = f"Here's the latest news on {format_crypto_name(crypto) if crypto else 'cryptocurrency'}:\n"
            for i, article in enumerate(articles[:3], 1):
                title = article.get("title", "No title")
                description = article.get("description", "No description")[:100] + "..." if len(article.get("description", "")) > 100 else article.get("description", "No description")
                url = article.get("url", "#")
                sentiment = article.get("sentiment", "neutral")
                response += f"{i}. **{title}** ({sentiment.capitalize()}) - {description} [Read more]({url})\n"
            return response
        return articles[0]["error"] if articles and articles[0].get("error") else f"Sorry, I couldn't fetch news for {format_crypto_name(crypto) if crypto else 'cryptocurrency'}. Check out our News page!"
    elif intent == "market":
        response = "Here's a quick crypto market overview:\n"
        for crypto in cryptos:
            price, change = fetch_live_price(crypto)
            if price is not None:
                change_text = f"(24hr: {round(float(change), 2)}%)" if change is not None else ""
                response += f"- {format_crypto_name(crypto)}: ${round(float(price), 2)} {change_text}\n"
        response += "For more details, check out our live prices or news sections!"
        return response
    elif intent == "portfolio":
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                holdings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            holdings = []
        
        if "add holding" in message_lower:
            match = re.search(r"add\s+(\d*\.?\d*)\s*(?:of)?\s*(btc|eth|bnb|ada|bitcoin|ethereum|binancecoin|cardano)\s*(?:at)?\s*\$?(\d*\.?\d*)", message_lower)
            if match:
                amount = float(match.group(1))
                crypto_input = match.group(2).lower()
                purchase_price = float(match.group(3))
                for crypto_name, aliases in crypto_aliases.items():
                    if crypto_input in aliases:
                        crypto = crypto_name
                        break
                else:
                    return "Invalid cryptocurrency. Use BTC, ETH, BNB, or ADA."
                holdings.append({"crypto": crypto, "amount": amount, "purchase_price": purchase_price})
                with open(PORTFOLIO_FILE, "w") as f:
                    json.dump(holdings, f)
                return f"Added {amount} {crypto.capitalize()} at ${purchase_price} to your portfolio!"
            return "Please specify like: 'Add 0.5 BTC at $80000'"
        elif "remove holding" in message_lower:
            match = re.search(r"remove\s+(\d*\.?\d*)\s*(?:of)?\s*(btc|eth|bnb|ada|bitcoin|ethereum|binancecoin|cardano)", message_lower)
            if match:
                amount = float(match.group(1))
                crypto_input = match.group(2).lower()
                for crypto_name, aliases in crypto_aliases.items():
                    if crypto_input in aliases:
                        crypto = crypto_name
                        break
                else:
                    return "Invalid cryptocurrency. Use BTC, ETH, BNB, or ADA."
                holdings = [h for h in holdings if not (h["crypto"] == crypto and h["amount"] == amount)]
                with open(PORTFOLIO_FILE, "w") as f:
                    json.dump(holdings, f)
                return f"Removed {amount} {crypto.capitalize()} from your portfolio!"
            return "Please specify like: 'Remove 0.5 BTC'"
        else:
            if not holdings:
                return "Your portfolio is empty. Try 'Add 0.5 BTC at $80000' or visit the Portfolio page."
            response = "Your portfolio:\n"
            total_value = 0
            for holding in holdings:
                price, _ = fetch_live_price(holding["crypto"])
                if price:
                    value = float(price * holding["amount"])
                    total_value += value
                    profit_loss = float((price - holding["purchase_price"]) * holding["amount"])
                    response += f"- {holding['amount']} {holding['crypto'].capitalize()}: ${round(float(value), 2)} (P/L: ${round(float(profit_loss), 2)})\n"
            response += f"Total value: ${round(float(total_value), 2)}"
            return response
    elif intent == "alert":
        try:
            with open(ALERTS_FILE, "r") as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []
        
        match = re.search(r"(alert|notify|watch)\s*(?:me)?\s*(?:if|when)?\s*(btc|eth|bnb|ada|bitcoin|ethereum|binancecoin|cardano)\s*(?:reaches|hits|goes)?\s*(above|below)?\s*\$?(\d*\.?\d*)", message_lower)
        if match:
            crypto_input = match.group(2).lower()
            direction = match.group(3) or "above"
            target_price = float(match.group(4))
            for crypto_name, aliases in crypto_aliases.items():
                if crypto_input in aliases:
                    crypto = crypto_name
                    break
            else:
                return "Invalid cryptocurrency. Use BTC, ETH, BNB, or ADA."
            alerts.append({"crypto": crypto, "target_price": target_price, "direction": direction})
            with open(ALERTS_FILE, "w") as f:
                json.dump(alerts, f)
            return f"Alert set for {crypto.capitalize()} {direction} ${target_price}! Check the Alerts page for details."
        return "Please specify like: 'Alert me if BTC reaches above $100000'"

    suggestions = []
    if crypto:
        suggestions.append(f"Try 'Current price of {format_crypto_name(crypto)}'")
        suggestions.append(f"Try 'Predict {format_crypto_name(crypto)} in 3 days'")
        suggestions.append(f"Try 'News about {format_crypto_name(crypto)}'")
        suggestions.append(f"Try 'Add 0.5 {format_crypto_name(crypto)} at $80000'")
        suggestions.append(f"Try 'Alert me if {format_crypto_name(crypto)} reaches above $100000'")
    if has_prediction_keyword or has_prediction_phrase:
        suggestions.append("Try 'Predict Bitcoin in 3 days'")
    if any(keyword in tokens for keyword in price_keywords):
        suggestions.append("Try 'Current price of Ethereum'")
    if any(keyword in tokens for keyword in history_keywords):
        suggestions.append("Try 'What was Cardano's price 7 days ago?'")
    if any(keyword in tokens for keyword in news_keywords):
        suggestions.append("Try 'Latest crypto news'")
    if any(keyword in tokens for keyword in market_keywords):
        suggestions.append("Try 'Crypto market overview'")
    if any(keyword in tokens for keyword in portfolio_keywords):
        suggestions.append("Try 'Add 0.5 BTC at $80000'")
    if any(keyword in tokens for keyword in alert_keywords):
        suggestions.append("Try 'Alert me if ETH reaches below $2000'")

    logger.debug(f"No intent matched for message: '{message}'")
    response = "Hmm, I'm not sure what you mean. Could you clarify?"
    if suggestions:
        response += " Here are some suggestions:\n- " + "\n- ".join(suggestions)
    else:
        response += " Try asking something like:\n- 'What will Bitcoin be in 3 days?'\n- 'Current price of Ethereum'\n- 'What was Cardano's price last week?'\n- 'Latest crypto news'\n- 'Add 0.5 BTC at $80000'\n- 'Alert me if BTC reaches $100000'\n- 'Crypto market overview'\nOr type 'help' for more options."
    return response

# Initialize data and models
for crypto in cryptos:
    logger.debug(f"Initializing data and model for {crypto}")
    data = fetch_data(crypto)
    if data is not None:
        logger.debug(f"Data fetched for {crypto}, rows: {len(data)}")
        historical_data[crypto] = data
        model = train_model(data, crypto)
        if model is not None:
            logger.debug(f"Model trained successfully for {crypto}")
            models[crypto] = model
        else:
            logger.error(f"Failed to train model for {crypto}")
    else:
        logger.error(f"Failed to fetch data for {crypto}")
    time.sleep(2)

if __name__ == '__main__':
    app.run(debug=True)