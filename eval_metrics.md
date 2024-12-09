### XGBoost Model instantiation
    # Split data into features and target
    X = processed_data.drop("Price", axis=1)
    y = processed_data["Price"]

    # Convert features to numeric, coercing errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)

    # Make predictions on training and test datasets
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)


### Model Metrics
- MAE (Train): 39713.16615449482
- RMSE (Train): 56380.9156186561
- R^2 (Train): 0.9003049239166963
- MAPE (Train): 12.544811118476954
- sMAPE (Train): 5.947660120611214
- MAE (Test): 58369.117824814886
- RMSE (Test): 85366.68455941485
- R^2 (Test): 0.772973053459807
- MAPE (Test): 17.60782056544349
- sMAPE (Test): 8.186809910648579

### Features
- Scraped data columns & shape:
  - Index(['Unnamed: 0', 'Locality', 'Zip Code', 'Type of Property',
       'Subtype of Property', 'Price', 'Type of Sale', 'Number of Rooms',
       'Livable Space (m2)', 'Fully Equipped Kitchen', 'Furnished',
       'Fireplace', 'Terrace', 'Terrace Area (m2)', 'Garden',
       'Garden Area (m2)', 'Swimming Pool', 'Surface of the Land (m2)',
       'Number of Facades', 'Construction Year', 'PEB',
       'Primary Energy Consumption (kWh/m2)', 'State of the Building', 'Url']
  - Data shape after loading: (34527, 24)
- Preprocessing, including (usually applied to specific column subsets predefined by me):
  - drop_unnecessary_columns
  - dropna
  - default to 0
  - drop useless columns
  - change 'Construction Year' to 'Building Age'
  - impute data for numeric values (KNNImputer)
  - remove outliers (IQR)
  - hot encode using pd.get_dummies
  - convert to numeric values
  - merge with geo data set (for municipality code) and precleaned fiscal data set (for prosperity index)
  - normalize numeric data
  - Data shape after preprocessing: (19919, 52)

### Efficiency
- Training time: 0.2241 seconds
- Inference time: 0.0072 seconds
