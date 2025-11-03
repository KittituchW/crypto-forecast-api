# AMLA_AT3_25544646_FastAPI

FastAPI microservice for **next-day cryptocurrency HIGH price prediction (t+2 horizon)**, developed as part of the **Advanced Machine Learning Application (AT3)** project at the University of Technology Sydney.

The API is hosted on **Render** with automated build and deployment from GitHub. Render manages dependency installation, environment setup, and HTTPS access.

**Live API:** [https://amla-at3-fastapi-latest.onrender.com](https://amla-at3-fastapi-latest.onrender.com)

This API provides real-time price prediction by fetching live data from the **Kraken OHLC API**, processing it into model-ready features, and returning the forecasted next-day HIGH price.

---

## Overview

The API predicts **tomorrow’s HIGH price** for a cryptocurrency using **yesterday’s features**. It combines live data retrieval, feature engineering, and machine learning inference into a single, deployable service.

**Key features:**

* Fetches up-to-date OHLC data from Kraken API
* Generates technical indicators and cyclical time features
* Loads a trained Lasso Regression model and scaler
* Returns next-day HIGH predictions as JSON

---

## Project Structure

```bash
AMLA_AT3_25544646_FastAPI/
│
├── app/
│   ├── __init__.py
│   ├── data_source.py        # Fetches OHLC data and builds features
│   ├── main.py               # FastAPI routes and endpoints
│   ├── utils.py              # Loads model bundle and handles prediction logic
│
├── models/
│   └── ETH/
│       ├── lasso_model.pkl           # Trained Lasso Regression model
│       ├── standard_scaler.pkl       # StandardScaler for feature normalization
│
├── Dockerfile                # Containerization setup for deployment
├── pyproject.toml            # Poetry dependency configuration
├── requirements.txt          # Backup dependency file
├── README.md
└── .gitignore
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/KittituchW/AMLA_at3_25544646_FastAPI.git
   cd AMLA_at3_25544646_FastAPI
   ```

2. **Install dependencies**

   Using **Poetry**:

   ```bash
   poetry install
   ```

   Or using **pip**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API locally**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will start at:
   `http://127.0.0.1:8000`

---

## API Endpoints

| Endpoint          | Method | Description                                                     |
| ----------------- | ------ | --------------------------------------------------------------- |
| `/`               | GET    | API overview and metadata                                       |
| `/health/`        | GET    | Returns `OK` if the API is healthy                              |
| `/predict/{pair}` | GET    | Returns predicted next-day HIGH price for the given crypto pair |

### Example Request

```bash
GET /predict/ETHUSD/
```

**Response:**

```json
{
  "predicted_next_day_high": 3952.36
}
```

### Health Check

You can check the service status using:

```bash
GET /health/
```

**Response:**

```
OK
```

---

## Model Details

* **Model:** Lasso Regression
* **Framework:** scikit-learn
* **Target Variable:** Next-day HIGH price (t+2 horizon)
* **Artifacts:**

  * `lasso_model.pkl`
  * `standard_scaler.pkl`

**Feature Set Includes:**

* Lagged prices (open, high, low, close)
* Moving averages (MA5, MA10, MA20)
* Exponential moving averages (EMA12, EMA26)
* Bollinger Bands
* Ichimoku indicators
* Cyclical month and weekday encodings
* Weekend flags

---

## Docker Deployment

**Build image:**

```bash
docker build -t amla-fastapi .
```

**Run container:**

```bash
docker run -p 8000:8000 amla-fastapi
```

Access locally at:
`http://localhost:8000`

---

## Author

**Kittituch Wongwatcharapaiboon**
Student ID: 25544646
University of Technology Sydney (UTS)
Subject: 36120 Advanced Machine Learning Application (AT3)
