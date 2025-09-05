In this project I have looked at stacked ML models as shown below to predict stocked prices. This was part of a wider university project.

`start.py` will download all required data from yahoo finance. `start.py` contains all config data as well such as the target ticker, External comparison tickers and file paths for saved predictions.

```
          Historical Prices ───► ARIMA ───┐
                                          │
           Time Series Data ───►  LSTM ───┤
                                          │
           Time Series Data ───►  GRU  ───┤
                                          │
  Technical Indicators & Features ─►  RF ─┤
                                          │
                                          ▼
                                Meta Learner: XGBoost
                                  (Final Prediction)
```
