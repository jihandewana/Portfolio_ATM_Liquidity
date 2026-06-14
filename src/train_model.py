import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def main():
    print("⏳ Loading unified feature matrix from SQLite database...")
    project_root = os.path.abspath(os.getcwd())
    database_file_path = os.path.join(project_root, 'atm_analytics.db')
    engine = create_engine(f"sqlite:///{database_file_path}")
    
    # 1. Extract data from memory table
    df = pd.read_sql_query("SELECT * FROM aggregated_atm_features;", con=engine)
    
    if len(df) == 0:
        print("⚠️ Database table is empty. Please run your aggregation script first!")
        return

    # 2. Re-align feature tracking columns in memory
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    df['Day_of_Week'] = df['Transaction_Date'].dt.dayofweek
    df['Day_of_Month'] = df['Transaction_Date'].dt.day
    df['Month'] = df['Transaction_Date'].dt.month
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    
    # Strictly sort sequentially by time to maintain chronological tracking integrity
    df = df.sort_values(by=['ATM_ID', 'Transaction_Date']).reset_index(drop=True)
    df['Cash_Out_Lag_1'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(1).fillna(0)
    df['Cash_Out_Lag_2'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(2).fillna(0)
    df['Cash_Out_Rolling_Mean_3d'] = df.groupby('ATM_ID')['Total_Cash_Out'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # 3. Step 8 Implementation: Separate target from modeling predictors
    feature_cols = [
        'Total_Transactions', 'Day_of_Week', 'Is_Weekend', 
        'Day_of_Month', 'Month', 'Cash_Out_Lag_1', 
        'Cash_Out_Lag_2', 'Cash_Out_Rolling_Mean_3d'
    ]
    
    X = df[feature_cols]
    y = df['Total_Cash_Out']

    # Chronological Train/Test split (80% historical training / 20% future evaluation)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"📊 Training Sample Rows : {X_train.shape[0]}")
    print(f"📊 Testing Validation Rows: {X_test.shape[0]}")

    # 4. Machine Learning Model Training
    print("\n🤖 Initializing Random Forest Liquidity Regressor...")
    # n_jobs=-1 utilizes all your computer's CPU cores for hyper-fast background training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("🏋️‍♂️ Training model on engineered time-series features...")
    model.fit(X_train, y_train)
    print("✅ Model training phase finished successfully!")

    # 5. Predictive Forecasting & Evaluation
    print("\n🔮 Generating cash demand forecasts on testing partition...")
    y_pred = model.predict(X_test)
    
    # Calculate performance errors
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print("\n📈 =====================================================")
    print("🤖 --- MACHINE LEARNING MODEL PERFORMANCE METRICS ---")
    print(f"🔹 Mean Absolute Error (MAE)       : {mae:.2f} Cash Units")
    print(f"🔹 Root Mean Squared Error (RMSE)   : {rmse:.2f} Cash Units")
    print("=========================================================")
    print("🎉 End-to-end predictive liquidity pipeline successfully executed!")

if __name__ == "__main__":
    main()