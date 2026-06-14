import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    print("⏳ Loading unified feature matrix from SQLite database...")
    project_root = os.path.abspath(os.getcwd())
    database_file_path = os.path.join(project_root, 'atm_analytics.db')
    engine = create_engine(f"sqlite:///{database_file_path}")
    
    df = pd.read_sql_query("SELECT * FROM aggregated_atm_features;", con=engine)
    
    # 1. Re-align feature tracking columns in memory
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    df['Day_of_Week'] = df['Transaction_Date'].dt.dayofweek
    df['Day_of_Month'] = df['Transaction_Date'].dt.day
    df['Month'] = df['Transaction_Date'].dt.month
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    
    df = df.sort_values(by=['ATM_ID', 'Transaction_Date']).reset_index(drop=True)
    df['Cash_Out_Lag_1'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(1).fillna(0)
    df['Cash_Out_Lag_2'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(2).fillna(0)
    df['Cash_Out_Rolling_Mean_3d'] = df.groupby('ATM_ID')['Total_Cash_Out'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # 2. Define final predictors matrix and target array
    feature_cols = [
        'Total_Transactions', 'Day_of_Week', 'Is_Weekend', 
        'Day_of_Month', 'Month', 'Cash_Out_Lag_1', 
        'Cash_Out_Lag_2', 'Cash_Out_Rolling_Mean_3d'
    ]
    
    X = df[feature_cols]
    y = df['Total_Cash_Out']

    # 3. Train the model on ALL historic records for maximum deployment capability
    print("🤖 Training the final model across the entire dataset...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    print("✅ Training complete.")

    # 4. Create paths for serialization outputs
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_output_path = os.path.join(models_dir, 'atm_rf_regressor.joblib')
    features_output_path = os.path.join(models_dir, 'feature_columns.joblib')

    # 5. Serialize and dump artifacts to the models directory
    print("\n💾 Exporting artifacts to the /models/ directory...")
    joblib.dump(final_model, model_output_path)
    joblib.dump(feature_cols, features_output_path)
    
    print("=========================================================")
    print(f"🎉 SUCCESS: Model saved at: {os.path.relpath(model_output_path)}")
    print(f"🎉 SUCCESS: Feature list saved at: {os.path.relpath(features_output_path)}")
    print("=========================================================")

if __name__ == "__main__":
    main()