**Money Talks — Cryptocurrency Price Prediction **

**Overview**

Money Talks is an intelligent cryptocurrency web application built using Flask and XGBoost to help users predict future prices of leading cryptocurrencies such as Bitcoin, Ethereum, Binance Coin, and Cardano. It also features live market data, historical price charts, sentiment-based news analysis, a chatbot assistant, and personal portfolio/alerts.

**Features**
- Accurate ML-based price prediction (XGBoost)
- Live crypto market prices (via CoinGecko API)
- Sentiment-based news section using NewsAPI
- NLP-powered chatbot assistant
- 
**Screenshots**
Include the following screenshots:
<img width="1920" height="1080" alt="Screenshot 2025-04-25 015154" src="https://github.com/user-attachments/assets/4ae28cdc-13f9-4c7e-8ca6-8f504bef55d7" />
<img width="1920" height="1080" alt="Screenshot 2025-04-25 015118" src="https://github.com/user-attachments/assets/ec0fe5d1-2f7b-4717-8002-b249b4e3819d" />
<img width="1920" height="1080" alt="Screenshot 2025-04-25 014117" src="https://github.com/user-attachments/assets/b61176e3-cf0a-4030-aaf6-dd98402cb664" />

**How It Works**
- ML Model: XGBoost Regressor trained on 365 days of historical prices.
- Features: Day index, lag1 (1-day price lag), lag7 (7-day lag), MA7 (7-day moving average).
- Prediction Range: 1–31 days
- Accuracy: ~88% to 99.7% based on real test logs.
**Tech Stack**
- Frontend: HTML, CSS, Bootstrap
- Backend: Python, Flask
- Machine Learning: XGBoost, Scikit-learn, Pandas, NumPy
- APIs Used: CoinGecko (price), NewsAPI (news)
- NLP: NLTK
- Logging: Python logging module
  
**Folder Structure**

├── app.py

├── templates/
       ├── index.html
       ├── about.html
       ├── news.html

├── static/
       ├── script.js
       ├── styles.css


**How to Run**
1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies with pip
4. Run the app using `python app.py`
5. Open your browser at http://127.0.0.1:5000
API Example
GET /predict?crypto=bitcoin&days=5
Example response:
{
"predicted_price": 93700.32,
"actual_price": 92318.23,
"accuracy_percentage": 98.50
}
**Future Enhancements**
- Add user accounts and dashboard
- Integrate Plotly/D3.js for interactive charts
- Email-based alert system
- Docker containerization
- GPT-4 chatbot assistant

**Author **
Shrishti Singh



