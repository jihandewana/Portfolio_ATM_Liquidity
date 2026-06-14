import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("⏳ Loading unified feature matrix from SQLite database...")
    project_root = os.path.abspath(os.getcwd())
    database_file_path = os.path.join(project_root, 'atm_analytics.db')
    engine = create_engine(f"sqlite:///{database_file_path}")
    
    df = pd.read_sql_query("SELECT * FROM aggregated_atm_features;", con=engine)
    
    # Re-align feature logic columns in memory
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

    feature_cols = [
        'Total_Transactions', 'Day_of_Week', 'Is_Weekend', 
        'Day_of_Month', 'Month', 'Cash_Out_Lag_1', 
        'Cash_Out_Lag_2', 'Cash_Out_Rolling_Mean_3d'
    ]
    
    X = df[feature_cols]
    y = df['Total_Cash_Out']

    # Chronological Train Split
    split_idx = int(len(df) * 0.80)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    print("🤖 Re-fitting Random Forest to extract feature weights...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Extract feature importances
    importances = model.feature_importances_
    
    # Create an organized DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 --- CALCULATED FEATURE IMPORTANCE WEIGHTS ---")
    print(importance_df.to_string(index=False))
    print("------------------------------------------------")

    # Build the Bar Chart Visual
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=importance_df, 
        x='Importance', 
        y='Feature', 
        hue='Feature',
        palette='viridis', 
        legend=False
    )
    
    plt.title('Random Forest Feature Importance Weights (Target: Total_Cash_Out)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Relative Predictive Weight Score', fontsize=12)
    plt.ylabel('Engineered Feature Predictors', fontsize=12)
    plt.tight_layout()
    
    print("🚀 Launching interactive feature importance window...")
    plt.show()

if __name__ == "__main__":
    main()