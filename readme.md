In this project I have looked at stacked ML models as shown below to predict stocked prices. This was part of a wider university project.
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
