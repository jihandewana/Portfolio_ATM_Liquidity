import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("⏳ Loading dataset from SQLite database for correlation check...")
    
    # 1. Establish database connection
    project_root = os.path.abspath(os.getcwd())
    database_file_path = os.path.join(project_root, 'atm_analytics.db')
    engine = create_engine(f"sqlite:///{database_file_path}")
    
    # 2. Extract data into Pandas
    df = pd.read_sql_query("SELECT * FROM aggregated_atm_features;", con=engine)
    
    if len(df) == 0:
        print("⚠️ The database table is empty! Run your aggregation pipeline first.")
        return

    # 3. Quick structural feature re-alignments for math scaling
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    df['Day_of_Week'] = df['Transaction_Date'].dt.dayofweek
    df['Month'] = df['Transaction_Date'].dt.month
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

    # Re-calculate lag/rolling contexts safely in memory
    df = df.sort_values(by=['ATM_ID', 'Transaction_Date']).reset_index(drop=True)
    df['Cash_Out_Lag_1'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(1).fillna(0)
    df['Cash_Out_Lag_2'] = df.groupby('ATM_ID')['Total_Cash_Out'].shift(2).fillna(0)
    df['Cash_Out_Rolling_Mean_3d'] = df.groupby('ATM_ID')['Total_Cash_Out'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # 4. Filter the column space down to modeling features
    feature_cols = [
        'Total_Cash_Out', 'Total_Transactions', 'Day_of_Week', 
        'Is_Weekend', 'Month', 'Cash_Out_Lag_1', 
        'Cash_Out_Lag_2', 'Cash_Out_Rolling_Mean_3d'
    ]
    
    corr_matrix = df[feature_cols].corr()

    # 5. Build and render the Heatmap
    print("🎨 Generating statistical correlation matrix...")
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        linewidths=0.5, 
        square=True,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Predictive Feature Correlation Matrix (Target: Total_Cash_Out)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    print("🚀 Launching interactive correlation window...")
    plt.show()
    print("👋 Window closed. Features are validated and ready for model split!")

if __name__ == "__main__":
    main()