# Crypto Forecast API
### Production-Ready ML Microservice for Cryptocurrency Price Prediction

FastAPI microservice for next-day cryptocurrency HIGH price forecasting (t+2 horizon). Designed and deployed as a production-style ML inference service using Docker and Render.

**Live API:** [https://amla-at3-fastapi-latest.onrender.com](https://amla-at3-fastapi-latest.onrender.com)

---

## 1. Overview
Crypto Forecast API is a containerized machine learning microservice that delivers short-term cryptocurrency forecasts using live market data. 

**The service:**
* Retrieves real-time OHLC data from the Kraken API.
* Performs feature engineering and preprocessing.
* Loads a trained regression model.
* Returns the predicted next-day HIGH price in JSON format.

This project demonstrates end-to-end ML deployment, including model packaging, API development, containerization, and cloud hosting.

---

## 2. Problem Definition
Cryptocurrency markets operate 24/7 and exhibit high volatility. To support short-term decision-making, this service forecasts the HIGH price two days ahead (t+2 horizon). 

**Why t+2?**
Because the HIGH price for a given day is only known after market close, predicting t+2 ensures:
* No information leakage.
* Realistic production constraints.
* Causally valid inference.

---

## 3. Architecture
The system follows a clean, modular ML deployment architecture:

1. **Data Ingestion**: Live market data is retrieved from Kraken OHLC API.
2. **Feature Engineering**: Pipeline generates technical indicators and lag features.
3. **Preprocessing**: Features are scaled using a pre-fitted `StandardScaler`.
4. **Inference**: A trained Lasso Regression model produces predictions.
5. **Delivery**: FastAPI returns results via a REST endpoint.
6. **Infrastructure**: Docker ensures environment consistency and Render hosts the container.



---

## 4. Project Structure
```text
crypto-forecast-api/
│
├── app/
│   ├── main.py             # FastAPI application and routes
│   ├── data_source.py      # Kraken API integration + feature builder
│   ├── utils.py            # Model loading and prediction logic
│   └── __init__.py
│
├── models/
│   └── ETH/
│       ├── lasso_model.pkl
│       └── standard_scaler.pkl
│
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API overview and metadata |
| `/health` | `GET` | Health check endpoint |
| `/predict/{pair}` | `GET` | Returns next-day HIGH prediction |

### Example Request
`GET /predict/ETHUSD/`

### Example Response
```json
{
  "predicted_next_day_high": 3952.36
}
```

## 6. Model Details

**Model Type**

Lasso Regression (scikit-learn)
Target: Next-day HIGH price (t+2 horizon)

**Feature Engineering**

The model uses engineered financial indicators including:

* *Lags:* Price features (open, high, low, close).
* *Momentum:* Moving averages (MA5, 10, 20) and EMA (12, 26).
* *Volatility:* Bollinger Bands.
* *Trend:* Ichimoku indicators.
* *Seasonality:* Cyclical month/weekday encodings and weekend indicators.

## 7. Local Development

**Clone Repository**
```bash
git clone https://github.com/KittituchW/crypto-forecast-api.git
cd crypto-forecast-api
```

**Install Dependencies**

Using Poetry:
```bash
poetry install
```

Using pip:
```bash
pip install -r requirements.txt
```

**Run API Locally**
```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000.

## 8. Docker Usage

**Build Image**
```bash
docker build -t crypto-forecast-api .
```

**Run Container**
```bash
docker run -p 8000:8000 crypto-forecast-api
```

## 9. Engineering Highlights

* **Time-aware logic:** Forecasting prevents data leakage.
* **Modular Design:** Clean separation between API, data, and logic.
* **Portability:** Production-ready Docker container.
* **CI/CD:** Automatic GitHub-based deployment on Render.
* **Live Integration:** Real-time external API connectivity.

## 10. Author

**Kittituch Wongwatcharapaiboon**

Master of Data Science and Innovation, University of Technology Sydney
Specialization: ML Engineering, Financial Time Series, Production ML Systems
